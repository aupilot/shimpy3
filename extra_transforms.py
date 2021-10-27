import random
import numpy as np
import cv2
from albumentations import DualTransform, CoarseDropout, GridDistortion
from albumentations.augmentations.crops import functional as F
from albumentations.augmentations.bbox_utils import denormalize_bbox, normalize_bbox, union_of_bboxes
from albumentations.augmentations.geometric import functional as FGeometric


class BBoxSafeRandomCrop(DualTransform):
    """Crop a random part of the input without loss of bboxes.
    Args:
        erosion_rate (float): erosion rate applied on input image height before crop.
        p (float): probability of applying the transform. Default: 1.
    Targets:
        image, mask, bboxes
    Image types:
        uint8, float32
    """

    def __init__(self, crop_height=0, crop_width=0, erosion_rate=0.0, always_apply=False, p=1.0):
        super(BBoxSafeRandomCrop, self).__init__(always_apply, p)
        self.erosion_rate = erosion_rate
        self.crop_height = crop_height
        self.crop_width = crop_width


    def apply(self, img, crop_height=0, crop_width=0, h_start=0, w_start=0, **params):
        return F.random_crop(img, self.crop_height, self.crop_width, h_start, w_start)

    def get_params_dependent_on_targets(self, params):
        img_h, img_w = params["image"].shape[:2]
        if len(params["bboxes"]) == 0:  # less likely, this class is for use with bboxes.
            erosive_h = int(img_h * (1.0 - self.erosion_rate))
            crop_height = img_h if erosive_h >= img_h else random.randint(erosive_h, img_h)
            return {
                "h_start": random.random(),
                "w_start": random.random(),
                "crop_height": crop_height,
                "crop_width": int(crop_height * img_w / img_h),
            }
        # get union of all bboxes
        x, y, x2, y2 = union_of_bboxes(
            width=img_w, height=img_h, bboxes=params["bboxes"], erosion_rate=self.erosion_rate
        )
        # find bigger region
        bx, by = x * random.random(), y * random.random()
        bx2, by2 = x2 + (1 - x2) * random.random(), y2 + (1 - y2) * random.random()
        bw, bh = bx2 - bx, by2 - by
        crop_height = img_h # if bh >= 1.0 else int(img_h * bh)
        crop_width = img_w  # if bw >= 1.0 else int(img_w * bw)
        h_start = np.clip(0.0 if bh >= 1.0 else by / (1.0 - bh), 0.0, 1.0)
        w_start = np.clip(0.0 if bw >= 1.0 else bx / (1.0 - bw), 0.0, 1.0)
        return {"h_start": h_start, "w_start": w_start, "crop_height": crop_height, "crop_width": crop_width}

    def apply_to_bbox(self, bbox, crop_height=0, crop_width=0, h_start=0, w_start=0, rows=0, cols=0, **params):
        return F.bbox_random_crop(bbox, self.crop_height, self.crop_width, h_start, w_start, rows, cols)

    @property
    def targets_as_params(self):
        return ["image", "bboxes"]

    def get_transform_init_args_names(self):
        return ("height", "width", "erosion_rate", "interpolation")


class Random256BBoxSafeCrop(DualTransform):
    """Crop a 256x256 part of the input and rescale it to some size without loss of bboxes.

    Args:
        height (int): height after crop and resize.
        width (int): width after crop and resize.
        erosion_rate (float): erosion rate applied on input image height before crop.
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes

    Image types:
        uint8, float32
    """
    def __init__(self, height, width, crop=256, test=False, interpolation=cv2.INTER_CUBIC, always_apply=False, p=1.0):
        super(Random256BBoxSafeCrop, self).__init__(always_apply, p)
        self.height = height
        self.width = width
        self.interpolation = interpolation
        self.test = test
        self.crop = crop

    def apply(self, img, xmin=0, ymin=0, xmax=0, ymax=0, interpolation=cv2.INTER_CUBIC, **params):
        # crop = F.random_crop(img, crop_height, crop_width, h_start, w_start)
        crop = F.crop(img, xmin, ymin, xmax, ymax)
        return crop #FGeometric.resize(crop, self.height, self.width, interpolation)

    def get_params_dependent_on_targets(self, params):
        img_h, img_w = params["image"].shape[:2]
        if len(params["bboxes"]) == 0:  # less likely, this class is for use with bboxes.
            y, x = int(random.random()*(1-self.crop/img_h)*img_h), int(random.random()*(1-self.crop/img_w)*img_w)
            return {"xmin": x, "ymin": y, "xmax": x+self.crop, "ymax": y+self.crop}
            # return {
            #     "h_start": random.random()*(1-self.crop/img_h),
            #     "w_start": random.random()*(1-self.crop/img_w),
            #     "crop_height": self.crop,
            #     "crop_width": self.crop,
            # }

        # get union of all bboxes
        # x, y, x2, y2 = union_of_bboxes(
        #     width=img_w, height=img_h, bboxes=params["bboxes"], erosion_rate=self.erosion_rate
        # )
        # we have only one box, so don't care
        (x, y, x2, y2, _) = params["bboxes"][0]

        # less randomness
        # bx = x + (random.random() * 0.4 - 0.2)
        # bxmin = np.clip(bx * img_w, 0, (img_w - self.crop))
        # by = y + (random.random() * 0.4 - 0.2)
        # bymin = np.clip(by * img_h, 0, (img_h - self.crop))

        # limit the shift according to the box size, otherwise we will get empty boxes when the box is small
        x_max_shift = (x2-x)*0.7
        y_max_shift = (y2-y)*0.7

        if not self.test:
            # we can either start from the end of the box or from the beginning
            xran = (random.random() * 0.2 - 0.1)
            xran = np.clip(xran, -x_max_shift, x_max_shift)
            if random.random() > 0.5:
                bx = x + xran
                bxmin = np.clip(bx * img_w, 0, (img_w - self.crop)-1)
            else:
                bx2 = x2 + xran
                bxmin = np.clip((bx2*img_w-self.crop), 0.0, (img_w - self.crop)-1)

            yran = (random.random() * 0.2 - 0.1)
            yran = np.clip(yran, -y_max_shift, y_max_shift)
            if random.random() > 0.5:
                by = y + yran
                bymin = np.clip(by * img_h, 0, (img_h - self.crop)-1)
            else:
                by2 = y2 + yran
                bymin = np.clip((by2*img_h-self.crop), 0.0, (img_h - self.crop)-1)
        else:
            # try to have the box in the centre
            bx = (x2+x)/2-self.crop/img_w/2
            by = (y2+y)/2-self.crop/img_h/2
            bxmin = np.clip(np.floor(bx * img_w), 0, (img_w - self.crop) - 1)
            bymin = np.clip(np.floor(by * img_h), 0, (img_h - self.crop) - 1)

        crop_width = self.crop
        crop_height = self.crop

        # bx = min(bx, (img_w-self.crop)/img_w)
        # by = min(bx, (img_h-self.crop)/img_h)

        # bx2, by2 = bx+self.crop/img_w, by+self.crop/img_h
        # bx2, by2 = x2 + (1 - x2) * random.random(), y2 + (1 - y2) * random.random()
        # bw, bh = bx2 - bx, by2 - by

        # crop_height = img_h if bh >= 1.0 else int(img_h * bh)
        # crop_width = img_w if bw >= 1.0 else int(img_w * bw)
        # h_start = np.clip(0.0 if bh >= 1.0 else by / (1.0 - bh), 0.0, 1.0)
        # w_start = np.clip(0.0 if bw >= 1.0 else bx / (1.0 - bw), 0.0, 1.0)
        # return {
        #     "h_start": h_start,
        #     "w_start": w_start,
        #     "crop_height": crop_height,
        #     "crop_width": crop_width}
        return {"xmin": int(bxmin), "ymin": int(bymin), "xmax": int(bxmin+crop_width), "ymax": int(bymin+crop_height)}

    def apply_to_bbox(self, bbox, xmin=0, ymin=0, xmax=0, ymax=0, rows=0, cols=0, **params):
        return F.bbox_crop(bbox, xmin, ymin, xmax, ymax, rows, cols)


    @property
    def targets_as_params(self):
        return ["image", "bboxes"]

    def get_transform_init_args_names(self):
        return ("height", "width", "crop", "test", "interpolation")


class CoarseDropoutBBoxes(CoarseDropout):
    def apply_to_bbox(self, bbox, xmin=0, ymin=0, xmax=0, ymax=0, rows=0, cols=0, **params):
        return bbox


class GridDistortionBBoxes(GridDistortion):
    def apply_to_bbox(self, bbox, xmin=0, ymin=0, xmax=0, ymax=0, rows=0, cols=0, **params):
        return bbox
