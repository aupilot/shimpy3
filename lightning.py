from numbers import Number
from typing import List
from functools import singledispatch
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
# from effdet import create_model
from fastcore.dispatch import typedispatch
from pytorch_lightning import LightningModule
from pytorch_lightning.core.decorators import auto_move_data
from ensemble_boxes import ensemble_boxes_wbf
# from data import get_valid_transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import create_model
import albumentations as A

def run_wbf(predictions, image_size=512, iou_thr=0.44, skip_box_thr=0.43, weights=None):
    bboxes = []
    confidences = []
    class_labels = []

    for prediction in predictions:
        boxes = [(prediction["boxes"] / image_size).tolist()]
        scores = [prediction["scores"].tolist()]
        labels = [prediction["classes"].tolist()]

        boxes, scores, labels = ensemble_boxes_wbf.weighted_boxes_fusion(
            boxes,
            scores,
            labels,
            weights=weights,
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr,
        )
        boxes = boxes * (image_size - 1)
        bboxes.append(boxes.tolist())
        confidences.append(scores.tolist())
        class_labels.append(labels.tolist())

    return bboxes, confidences, class_labels



# def get_valid_transforms(target_img_size=512):
#     return A.Compose(
#         [
#             A.Resize(height=target_img_size, width=target_img_size, p=1),
#             A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#             ToTensorV2(p=1),
#         ],
#         p=1.0,
#         bbox_params=A.BboxParams(
#             format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
#         ),
#     )


class EfficientDetModel(LightningModule):
    def __init__(
            self,
            num_classes=35,
            img_size=512,
            prediction_confidence_threshold=0.2,
            learning_rate=0.0002,
            wbf_iou_threshold=0.44,
            skip_box_thr=0.1,
            inference_transforms=None,
            model_architecture='tf_efficientdet_d0',
            # model_architecture='tf_efficientnetv2_l',
            len_data_loader=0, epochs=0, batch_size=0
    ):
        super().__init__()

        self.len_data_loader = len_data_loader
        self.bs = batch_size
        self.epochs = epochs
        # if inference_transforms is None:
        #     inference_transforms = get_valid_transforms(target_img_size=img_size)
        self.img_size = img_size
        self.model = create_model(
            num_classes, img_size, architecture=model_architecture
        )
        self.skip_box_thr = skip_box_thr
        self.prediction_confidence_threshold = prediction_confidence_threshold
        self.lr = learning_rate
        self.wbf_iou_threshold = wbf_iou_threshold
        # self.inference_tfms = inference_transforms

    # @auto_move_data
    def forward(self, images, targets):
        return self.model(images, targets)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.2, patience=5)
        return {
           'optimizer': optimizer,
           'lr_scheduler': scheduler,
           'monitor': 'valid_loss'
        }

    # def configure_optimizers(self):
    #     optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
    #     scheduler = {
    #         'scheduler': torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.005, steps_per_epoch=self.len_data_loader//self.bs, epochs=self.epochs),
    #         'interval': 'step'  # called after each training step
    #     }
    #     return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        images, annotations = batch
        losses = self.model(images, annotations)

        # logging_losses = {
        #     "class_loss": losses["class_loss"].detach(),
        #     "box_loss": losses["box_loss"].detach(),
        # }

        self.log("train_loss", losses["loss"], on_step=True, on_epoch=True,
                 prog_bar=False,
                 logger=True)
        self.log(
            "train_class_loss", losses["class_loss"], on_step=True, on_epoch=True, prog_bar=True,
            logger=True
        )
        self.log("train_box_loss", losses["box_loss"], on_step=True, on_epoch=True,
                 prog_bar=False,
                 logger=True)

        return losses['loss']

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        # images, annotations, targets, image_ids = batch
        images, annotations = batch
        outputs = self.model(images, annotations)

        detections = outputs["detections"]

        # batch_predictions = {
        #     "predictions": detections,
        #     # "targets": targets,
        #     # "image_ids": image_ids,
        # }

        logging_losses = {
            "class_loss": outputs["class_loss"].detach(),
            "box_loss": outputs["box_loss"].detach(),
        }

        self.log("valid_loss", outputs["loss"], on_step=True, on_epoch=True, prog_bar=False,
                 logger=True, sync_dist=True)
        self.log(
            "valid_class_loss", logging_losses["class_loss"], on_step=True, on_epoch=True,
            prog_bar=True, logger=True, sync_dist=True
        )
        self.log("valid_box_loss", logging_losses["box_loss"], on_step=True, on_epoch=True,
                 prog_bar=False, logger=True, sync_dist=True)

        # return {'loss': outputs["loss"], 'batch_predictions': batch_predictions}
        return {'loss': outputs["loss"]}

    def predicts_step(self, batch):
        return self(batch)

    # @typedispatch
    # def predict(self, images: List):
    #     """
    #     For making predictions from images
    #     Args:
    #         images: a list of PIL images
    #
    #     Returns: a tuple of lists containing bboxes, predicted_class_labels, predicted_class_confidences
    #
    #     """
    #     image_sizes = [(image.size[1], image.size[0]) for image in images]
    #     images_tensor = torch.stack(
    #         [
    #             self.inference_tfms(
    #                 image=np.array(image, dtype=np.float32),
    #                 labels=np.ones(1),
    #                 bboxes=np.array([[0, 0, 1, 1]]),
    #             )["image"]
    #             for image in images
    #         ]
    #     )
    #
    #     return self._run_inference(images_tensor, image_sizes)

    @typedispatch
    def predict(self, images_tensor: torch.Tensor):
        """
        For making predictions from tensors returned from the model's dataloader
        Args:
            images_tensor: the images tensor returned from the dataloader

        Returns: a tuple of lists containing bboxes, predicted_class_labels, predicted_class_confidences

        """
        if images_tensor.ndim == 3:
            images_tensor = images_tensor.unsqueeze(0)
        if (
                images_tensor.shape[-1] != self.img_size
                or images_tensor.shape[-2] != self.img_size
        ):
            raise ValueError(
                f"Input tensors must be of shape (N, 3, {self.img_size}, {self.img_size})"
            )

        num_images = images_tensor.shape[0]
        image_sizes = [(self.img_size, self.img_size)] * num_images

        return self._run_inference(images_tensor, image_sizes)

    def _run_inference(self, images_tensor, image_sizes):
        dummy_targets = self._create_dummy_inference_targets(
            num_images=images_tensor.shape[0]
        )

        detections = self.model(images_tensor.to(self.device), dummy_targets)[
            "detections"
        ]
        (
            predicted_bboxes,
            predicted_class_confidences,
            predicted_class_labels,
        ) = self.post_process_detections(detections)

        scaled_bboxes = self.__rescale_bboxes(
            predicted_bboxes=predicted_bboxes, image_sizes=image_sizes
        )

        return scaled_bboxes, predicted_class_labels, predicted_class_confidences

    def _create_dummy_inference_targets(self, num_images):
        dummy_targets = {
            "bbox": [
                torch.tensor([[0.0, 0.0, 0.0, 0.0]], device=self.device)
                for i in range(num_images)
            ],
            "cls": [torch.tensor([1.0], device=self.device) for i in range(num_images)],
            "img_size": torch.tensor(
                [(self.img_size, self.img_size)] * num_images, device=self.device
            ).float(),
            "img_scale": torch.ones(num_images, device=self.device).float(),
        }

        return dummy_targets

    def post_process_detections(self, detections):
        predictions = []
        for i in range(detections.shape[0]):
            predictions.append(
                self._postprocess_single_prediction_detections(detections[i])
            )

        predicted_bboxes, predicted_class_confidences, predicted_class_labels = run_wbf(
            predictions, image_size=self.img_size, iou_thr=self.wbf_iou_threshold, skip_box_thr=self.skip_box_thr
        )

        return predicted_bboxes, predicted_class_confidences, predicted_class_labels

    def _postprocess_single_prediction_detections(self, detections):
        boxes = detections.detach().cpu().numpy()[:, :4]
        scores = detections.detach().cpu().numpy()[:, 4]
        classes = detections.detach().cpu().numpy()[:, 5]
        indexes = np.where(scores > self.prediction_confidence_threshold)[0]
        boxes = boxes[indexes]

        return {"boxes": boxes, "scores": scores[indexes], "classes": classes[indexes]}

    def __rescale_bboxes(self, predicted_bboxes, image_sizes):
        scaled_bboxes = []
        for bboxes, img_dims in zip(predicted_bboxes, image_sizes):
            im_h, im_w = img_dims

            if len(bboxes) > 0:
                scaled_bboxes.append(
                    (
                            np.array(bboxes)
                            * [
                                im_w / self.img_size,
                                im_h / self.img_size,
                                im_w / self.img_size,
                                im_h / self.img_size,
                            ]
                    ).tolist()
                )
            else:
                scaled_bboxes.append(bboxes)

        return scaled_bboxes
