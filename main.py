from data import ShimpyDataModule, class_bins
from lightning import EfficientDetModel
import pandas as pd
from pytorch_lightning import Trainer
from effdet.config.model_config import efficientdet_model_param_dict
from pytorch_lightning.callbacks import ModelCheckpoint


# https://medium.com/data-science-at-microsoft/training-efficientdet-on-custom-data-with-pytorch-lightning-using-an-efficientnetv2-backbone-1cdf3bd7921f
# https://gist.github.com/Chris-hughes10/73628b1d8d6fc7d359b3dcbbbb8869d7
# timm.list_models('tf_efficientnetv2_*')           # list v2
# list(efficientdet_model_param_dict.keys())[::3]   # list v1

# dataset: drop frames with prop < 0.3 instead of 0.2
# added erossion rate 0.2 instead of 0
# more epochs (36 inst of 30)

N_FOLDS = 4

if __name__ == '__main__':

    image_size = [512, 512]
    checkpoint_callback = ModelCheckpoint(monitor="valid_class_loss", save_top_k=3)

    for f in range(N_FOLDS):
        dm = ShimpyDataModule(f, folds=N_FOLDS, batch_size=3, image_size=image_size, num_workers=0)

        model = EfficientDetModel(
            # model_architecture='tf_efficientdet_d0',
            model_architecture='tf_efficientnetv2_l',
            num_classes=len(class_bins),
            img_size=image_size[0],

        )

        trainer = Trainer(gpus=[0], max_epochs=36, num_sanity_val_steps=4, callbacks=[checkpoint_callback])
        trainer.fit(model, dm)