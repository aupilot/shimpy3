from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from data import ShimpyDataModule, class_bins
from lightning import EfficientDetModel
import pandas as pd
from pytorch_lightning import Trainer

# https://medium.com/data-science-at-microsoft/training-efficientdet-on-custom-data-with-pytorch-lightning-using-an-efficientnetv2-backbone-1cdf3bd7921f



N_FOLDS = 1
BS = 16
EPOCHS = 26

if __name__ == '__main__':
    image_size = [512, 512]

    for f in range(N_FOLDS):
        dm = ShimpyDataModule(f, folds=N_FOLDS, batch_size=BS, image_size=image_size, num_workers=0)
        dm.setup()

        checkpoint_callback = ModelCheckpoint(monitor="valid_class_loss", save_top_k=3)
        lr_monitor = LearningRateMonitor(logging_interval='step')

        model = EfficientDetModel(
            model_architecture='tf_efficientdet_d0',
            num_classes=len(class_bins),
            img_size=image_size[0],
            learning_rate=0.001, #0.0002,
            len_data_loader=len(dm.train_dataset),
            epochs=EPOCHS,
            batch_size=BS,
        )

        trainer = Trainer(gpus=[0], precision=16, max_epochs=EPOCHS, num_sanity_val_steps=4, callbacks=[checkpoint_callback, lr_monitor])
        trainer.fit(model, dm)