from typing import Optional
import cv2
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
import torch
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedKFold
import albumentations as A
from albumentations.pytorch import ToTensorV2

from extra_transforms import BBoxSafeRandomCrop, Random256BBoxSafeCrop, CoarseDropoutBBoxes, GridDistortionBBoxes

input_dir = r"E:/Chimpact/"
test_image_dir = r"test_images/"
train_images_dir  = r"train_images_multi/"
# test_image_dir = "tmp_images\\"   # @@@@@@@@@@@@@@@@@@@@@@@@@@
# input_dir = "/Users/kir/Datasets/Shimpact/"     # don't use the ~ shortcut for /users/kir !

labels_file = input_dir + "train_labels_new.csv"
meta_file = input_dir + "train_meta_new.csv"

test_meta_file = input_dir + "test_metadata.csv"
# test_meta_file = input_dir + "train_metadata.csv" # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@

class_bins = np.array([  0,   5,  10,  15,  20,  25,  30,  35,  40,  45,  50,  55,  60,
                    65,  70,  75,  80,  90, 100, 110, 120, 130, 135, 140, 150, 160,
                    170, 180, 190, 200, 210, 220, 230, 240, 250])


class ShimpyDataModule(LightningDataModule):

    def __init__(self, fold_no: 0, folds: 2, image_size=None, batch_size: int=32, num_workers=4, random=True):
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.fold_no = fold_no
        self.folds = folds
        self.image_size = image_size

    # def prepare_data(self):
    #     # called only on 1 GPU
    #     download_dataset()

    def setup(self, stage: Optional[str] = None, random=True):

        if self.folds > 1:
            test_fold = self.fold_no + 1
            if test_fold >= self.folds:
                test_fold = 0

            train_datasets = []
            for i in range(self.folds):
                if i is not test_fold:
                    new_ds = ShimpyDataset(fold_no=i, folds=self.folds, transforms=None, random=random, image_size=self.image_size)
                    train_datasets.append(new_ds)

            self.train_dataset = ConcatDataset(train_datasets)
            self.valid_dataset = ShimpyDataset(fold_no=test_fold, folds=self.folds, transforms=None, random=random, image_size=self.image_size)
            indices = torch.arange(len(self.valid_dataset)//2)       # lets cut the valid dataset twice to speed up
            self.valid_dataset = torch.utils.data.Subset(self.valid_dataset, indices)
        else:
            self.train_dataset = ShimpyDataset(fold_no=0, folds=1, transforms=None, random=random, image_size=self.image_size)
            indices = torch.randint_like(torch.arange(len(self.train_dataset)), 0, len(self.train_dataset))[0: len(self.train_dataset)//6]
            self.valid_dataset = torch.utils.data.Subset(self.train_dataset, indices)

        self.test_dataset = ShimpyTestDataset(image_size=[512, 512])

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          pin_memory=True,
                          drop_last=True,
                          num_workers=self.num_workers,
                          collate_fn=self.collate_fn,
                          )

    def val_dataloader(self):
        return DataLoader(self.valid_dataset,
                          batch_size=self.batch_size,
                          pin_memory=True,
                          shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=self.collate_fn,
                          )

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.collate_test_fn)

    # @staticmethod
    def collate_fn(self, batch):
        images, targets = tuple(zip(*batch))
        images = torch.stack(images)
        images = images.permute(0, 3, 1, 2)

        boxes = [target["bboxes"].float() for target in targets]
        labels = [target["labels"].float() for target in targets]
        img_size = torch.tensor([self.image_size for target in targets]).float()
        img_scale = torch.tensor([1.0 for target in targets]).float()
        img_ids = [target["img_ids"] for target in targets]

        annotations = {
            "bbox": boxes,
            "cls": labels,
            "img_size": img_size,
            "img_scale": img_scale,
            "img_file": img_ids,
        }

        return images, annotations

    def collate_test_fn(self, batch):

        images, targets = tuple(zip(*batch))
        images = torch.stack(images)
        images = images.permute(0, 3, 1, 2)

        boxes = [target["bboxes"].float() for target in targets]
        labels = [target["labels"].float() for target in targets]
        img_ids = [target["img_ids"] for target in targets]
        img_size = torch.tensor([self.image_size for target in targets]).float()
        img_scale = torch.tensor([1.0 for target in targets]).float()

        annotations = {
            "bbox": boxes,
            "cls": labels,
            "img_size": img_size,
            "img_scale": img_scale,
            "img_file": img_ids,
        }

        return images, annotations


# load the labels and metadata. Split into folds taking into account the place the video was taken
class ShimpyDataset(Dataset):

    def __init__(self, fold_no=0, folds=2, transforms=None, random=True, image_size=None, crop_size=256 ):
        super().__init__()
        self.transforms = transforms
        # self.test = test
        self.crop_size = crop_size
        self.image_size = image_size
        self.image_dir = train_images_dir
        # if test:
        #     self.image_dir = "train_images/"
        #     # self.image_dir = "test_images/"
        # else:
        #     self.image_dir = "train_images/"

        meta = pd.read_csv(meta_file, skipinitialspace=True)
        labels = pd.read_csv(labels_file, skipinitialspace=True)


        # TODO: instead of dropping set class 0 and box to 0..1, 0..1. But then we need to shift classes (no +1)

        # drop all frames with nans
        idx_nans = meta[pd.isnull(meta['x1'])].index
        # labels.drop(idx_nans, inplace=True)
        # meta.drop(idx_nans, inplace=True)
        # we don't drop - we replace bbox withg full screen instead
        meta.at[idx_nans, ['x1', 'y1']] = 0.0
        meta.at[idx_nans, ['x2', 'y2']] = 1.0

        # TODO: fine tune what we should drop @@@@@@@@@@@@@@@@
        # drop all frames with probability less than 0.03 or 0.05?
        # e.g. aiwa.mp4,0,0.61429566,0.46316695,0.9058581,0.8004757,0.03351803,moyen_bafing,fat is stil good
        # ajxx.avi,1,0.0701513,0.0,0.9299861,0.93549573,0.011272797,tai,tgq - no good
        # amwt.avi,2,0.103925645,0.0,0.8923694,0.7745187,0.022369707,tai,wqz - no good
        # apob.mp4,50,0.0033145845,0.19179106,0.11817341,0.6443564,0.044571202,moyen_bafing,fat - no good

        df_folds = meta[['video_id']].copy()
        df_folds.loc[:, 'frame_count'] = 1                  # добавили столбец frame_count == 1 везде
        df_folds = df_folds.groupby('video_id').count()     # посчитали сколько строк с одного видео
        df_folds.loc[:, 'source'] = meta[['video_id', 'site_id']].groupby('video_id').min()['site_id']
        df_folds.loc[:, 'stratify_group'] = np.char.add(
            df_folds['source'].values.astype(str),
            df_folds['frame_count'].apply(lambda x: f'_{x // 1}').values.astype(str)
        )
        df_folds.loc[:, 'fold'] = 0

        if folds > 1:
            skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=73)
            for fold_number, (train_index, val_index) in enumerate(
                    skf.split(X=df_folds.index, y=df_folds['stratify_group'])):
                df_folds.loc[df_folds.iloc[val_index].index, 'fold'] = fold_number

            mask = df_folds[df_folds['fold'] == fold_no].index

            self.fold_meta = None
            for mmm in mask:
                if self.fold_meta is None:
                    self.fold_meta = meta[meta['video_id'] == mmm].copy()
                    self.fold_labels = labels[labels['video_id'] == mmm].copy()
                else:
                    self.fold_meta = self.fold_meta.append(meta[meta['video_id'] == mmm])
                    self.fold_labels = self.fold_labels.append(labels[labels['video_id'] == mmm])

            self.fold_labels.reset_index(inplace=True)
            self.fold_meta.reset_index(inplace=True)

        else:
            self.fold_labels = labels.copy()
            self.fold_meta = meta.copy()

        # shuffle if not test
        if random:
            shuffled_idx = np.random.permutation(self.fold_labels.index)
            self.fold_labels = self.fold_labels.reindex(shuffled_idx)
            self.fold_meta = self.fold_meta.reindex(shuffled_idx)

        self.transform = A.Compose([
            # A.RandomCrop(height=256,width=256),
            # BBoxSafeRandomCrop(crop_width=256, crop_height=256, erosion_rate=0.0),
            # A.RandomSizedBBoxSafeCrop(width=self.image_size[1], height=self.image_size[0], erosion_rate=0.2 , interpolation=cv2.INTER_CUBIC),
            Random256BBoxSafeCrop(width=self.image_size[1], height=self.image_size[0], crop=self.crop_size, test=False),
            A.Resize(height=self.image_size[0], width=self.image_size[1], interpolation=cv2.INTER_LINEAR),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.GaussNoise(var_limit=(0.01, 0.2), p=0.2),
            # GridDistortionBBoxes(distort_limit=0.2, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0.5, p=0.2),
            CoarseDropoutBBoxes(max_height=28,
                                max_width=28,
                                min_height=8,
                                min_width=8,
                                p=0.25),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))

    def load_image_and_box(self, index):

        if isinstance(index, torch.Tensor):
            row = self.fold_labels.iloc[index.item()]
        else:
            row = self.fold_labels.iloc[index]

        image_file = "img_" + str.split(row.video_id, '.')[0] + f"_{row.time:04d}.png"

        image = cv2.imread(input_dir+self.image_dir+image_file, cv2.IMREAD_COLOR)
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        except:
            print("no image")
        image /= 255.0

        # return distance as a class label
        ## class labels start from 1 and the background class = -1
        dist = np.where(class_bins == int(row.distance*10))[0]+1
        if dist.size == 0:
            raise Exception('Something wrong with data! The distance does not fit to bins')

        details = self.fold_meta[(self.fold_meta['video_id'] == row.video_id) & (self.fold_meta['time'] == row.time)]

        boxes = details[['x1', 'y1', 'x2', 'y2']].values    # swap x/y to match torch expectations
        boxes[:, 0] *= image.shape[1]
        boxes[:, 2] *= image.shape[1]
        boxes[:, 1] *= image.shape[0]
        boxes[:, 3] *= image.shape[0]

        transformed = self.transform(image=image, bboxes=boxes, category_ids=dist)

        return transformed, image_file

    def __getitem__(self, idx: int):
        sample, image_file = self.load_image_and_box(idx)
        target = {'labels': torch.tensor(np.array(sample['category_ids'],dtype='int32')),
                  'bboxes': torch.tensor(np.array(sample['bboxes']).astype('float32')),
                  'img_ids': image_file
                  }

        return torch.tensor(sample['image']),target

    def __len__(self) -> int:
        return len(self.fold_labels)


class ShimpyTestDataset(Dataset):

    def __init__(self, transforms=None, image_size=None, crop_size=256):
        super().__init__()
        self.image_size = image_size
        self.image_dir  = test_image_dir
        self.crop_size = crop_size

        meta = pd.read_csv(test_meta_file, skipinitialspace=True)
        idx_nans = meta[pd.isnull(meta['x1'])].index
        meta.iloc[idx_nans, 2] = 0.0
        meta.iloc[idx_nans, 3] = 0.0
        meta.iloc[idx_nans, 4] = 1.0
        meta.iloc[idx_nans, 5] = 1.0

        df_folds = meta[['video_id']].copy()
        df_folds.loc[:, 'frame_count'] = 1                  # добавили столбец frame_count == 1 везде
        df_folds = df_folds.groupby('video_id').count()     # посчитали сколько строк с одного видео
        df_folds.loc[:, 'source'] = meta[['video_id', 'site_id']].groupby('video_id').min()['site_id']
        df_folds.loc[:, 'stratify_group'] = np.char.add(
            df_folds['source'].values.astype(str),
            df_folds['frame_count'].apply(lambda x: f'_{x // 1}').values.astype(str)
        )
        df_folds.loc[:, 'fold'] = 0

        self.fold_meta = meta.copy()

        # TODO: perhaps use larger images?
        # we use the test dataset boxes provided top crop the animal
        if transforms is None:
            self.transform = A.Compose([
                # A.RandomCrop(height=256,width=256),
                # BBoxSafeRandomCrop(crop_width=256, crop_height=256, erosion_rate=0.0),
                # A.RandomSizedBBoxSafeCrop(width=self.image_size[1], height=self.image_size[0], erosion_rate=0.2 , interpolation=cv2.INTER_CUBIC),
                A.PadIfNeeded(min_width=640, min_height=360, border_mode=cv2.BORDER_CONSTANT, value=0.5),   # some images are smaller than 360. We will pad them
                # A.PadIfNeeded(min_width=1280, min_height=720, border_mode=cv2.BORDER_CONSTANT, value=0.5),   # hi-res version
                Random256BBoxSafeCrop(width=self.image_size[1], height=self.image_size[0], crop=self.crop_size, test=True),
                A.Resize(height=512, width=512, interpolation=cv2.INTER_LINEAR),     # when using 256x256 crop we need to upscale for the net input
                # A.HorizontalFlip(p=0.5),
                # A.RandomBrightnessContrast(p=0.2),
                # ToTensorV2(p=1),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))
        else:
            self.transforms = transforms

    def load_image_and_box(self, index):

        if isinstance(index, torch.Tensor):
            row = self.fold_meta.iloc[index.item()]
        else:
            row = self.fold_meta.iloc[index]

        video = str.split(row.video_id, '.')
        image_file = "img_" + video[0] + '_' + video[1] + f"_{row.time:04d}.png"

        image = cv2.imread(input_dir+self.image_dir+image_file, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        # details = self.fold_meta[(self.fold_meta['video_id'] == row.video_id) & (self.fold_meta['time'] == row.time)]

        boxes = np.expand_dims(row[['x1', 'y1', 'x2', 'y2']].values, axis=0)    # swap x/y to match torch expectations
        boxes[:, 0] *= image.shape[1]
        boxes[:, 2] *= image.shape[1]
        boxes[:, 1] *= image.shape[0]
        boxes[:, 3] *= image.shape[0]

        transformed = self.transform(image=image, bboxes=boxes, category_ids=np.zeros((1,)))

        return transformed, image_file
        # return image, boxes, dist

    def __getitem__(self, idx: int):
        sample, image_file = self.load_image_and_box(idx)

        target = {'labels': torch.tensor(np.array(sample['category_ids'],dtype='int32')),
                  'bboxes': torch.tensor(np.array(sample['bboxes']).astype('float32')),
                  'img_ids': image_file,
                  }

        return torch.tensor(sample['image']),target

    def __len__(self) -> int:
        return len(self.fold_meta)


def test_train_dataset():
    ds = ShimpyDataset(folds=2, fold_no=0, image_size=[512, 512])
    print(len(ds))
    # sample = ds[45]
    # [*boxes] = sample['bboxes']
    # image = sample['image']

    # [*boxes] = sample[1]
    # image = sample[0]
    for aa in range(150):
        image, target = ds[aa]
        image = image.numpy()

        # print probability
        frame = int(target['img_ids'].split('_')[2].split('.')[0])
        video = target['img_ids'].split('_')[1]
        print(f"Prob: {ds.fold_meta[(ds.fold_meta['time'] == frame) & (ds.fold_meta['video_id'].str.contains(video))]['probability'].item()}")

        # for box in target['boxes']:
        box = target['bboxes']
        cv2.rectangle(image, (int(box[0,0]), int(box[0,1])), (int(box[0,2]), int(box[0,3])), (0, 1, 0), 2)

        cv2.imshow(f"class {target['labels'][0].numpy()}", image)
        # cv2.imshow(f"class {sample['category_ids'][0].item()}", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def test_lit_datamodule():
    ldm = ShimpyDataModule(0,2, batch_size=4, image_size=[512, 512])
    ldm.setup()
    dataloader = ldm.train_dataloader()
    image, target = next(iter(dataloader))
    print(image.shape)

def test_test_dataset():
    ds = ShimpyTestDataset(image_size=[512, 512])
    print(len(ds))
    # sample = ds[45]
    # [*boxes] = sample['bboxes']
    # image = sample['image']

    # [*boxes] = sample[1]
    # image = sample[0]

    for aa in range(10):
        image, target = ds[aa]
        image = image.numpy()
        # for box in target['boxes']:
        box = target['bboxes']
        cv2.rectangle(image, (int(box[0,0]), int(box[0,1])), (int(box[0,2]), int(box[0,3])), (0, 1, 0), 2)

        cv2.imshow(f"class {target['labels'][0].numpy()}", image)
        # cv2.imshow(f"class {sample['category_ids'][0].item()}", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # test_lit_datamodule()
    test_train_dataset()
    # test_test_dataset()
