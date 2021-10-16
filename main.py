from data import ShimpyDataModule, class_bins
from lightning import EfficientDetModel
import pandas as pd
from pytorch_lightning import Trainer

# https://medium.com/data-science-at-microsoft/training-efficientdet-on-custom-data-with-pytorch-lightning-using-an-efficientnetv2-backbone-1cdf3bd7921f

# dataset: drop frames with prop < 0.3 instead of 0.2
# added erossion rate 0.2 instead of 0
# more epochs (36 inst of 30)

N_FOLDS = 1

if __name__ == '__main__':

    image_size = [512, 512]

    for f in range(N_FOLDS):
        dm = ShimpyDataModule(f, folds=N_FOLDS, batch_size=8, image_size=image_size, num_workers=0)

        model = EfficientDetModel(
            model_architecture='tf_efficientdet_d0',
            num_classes=len(class_bins),
            img_size=image_size[0]
        )

        trainer = Trainer(gpus=[0], max_epochs=36, num_sanity_val_steps=4)
        trainer.fit(model, dm)