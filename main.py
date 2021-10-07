from data import ShimpyDataModule, class_bins
from lightning import EfficientDetModel
import pandas as pd
from pytorch_lightning import Trainer

# https://medium.com/data-science-at-microsoft/training-efficientdet-on-custom-data-with-pytorch-lightning-using-an-efficientnetv2-backbone-1cdf3bd7921f


if __name__ == '__main__':

    image_size = [512, 512]

    dm = ShimpyDataModule(0, 2, batch_size=8, image_size=image_size, num_workers=0)

    model = EfficientDetModel(
        model_architecture='tf_efficientdet_d0',
        num_classes=len(class_bins),
        img_size=512
    )

    trainer = Trainer(gpus=[0], max_epochs=24, num_sanity_val_steps=10)
    trainer.fit(model, dm)