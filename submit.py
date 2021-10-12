from effdet import create_model
import torch
from effdet.soft_nms import batched_soft_nms

from data import ShimpyDataModule, class_bins
from lightning import EfficientDetModel
import pandas as pd
from pytorch_lightning import Trainer
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2


chp = r"C:\Users\kir\Documents\Python\Shimpy3\lightning_logs\version_27\checkpoints\epoch=23-step=13319.ckpt"
name = 'm1-fold0'
# chp = r"C:\Users\kir\Documents\Python\Shimpy3\lightning_logs\version_28\checkpoints\epoch=29-step=16949.ckpt"
# name = 'm1-fold1'

if __name__ == '__main__':
    device = 'cuda'
    image_size = [512, 512]

    dm = ShimpyDataModule(0, 2, batch_size=10, image_size=image_size, num_workers=0)
    dm.setup()

    model = EfficientDetModel.load_from_checkpoint(chp)
    model.eval()
    model.wbf_iou_threshold = 0.2
    model.skip_box_thr = 0.02

    model.to(device)
    data_aslist = []

    for images, targets in dm.predict_dataloader():
        a = model.predict(images.to(device))
        for i, image in enumerate(images):
            img_file = targets['img_file'][i]
            cls = a[1][i]
            if len(cls) > 0:
                dists = [class_bins[int(c-1)] for c in cls]
                dist = np.mean(dists)/10
                strongest_box = np.argmax(a[2][i])
                box = a[0][i][strongest_box]
                img_np = image.permute(1, 2, 0).numpy().copy()
                cv2.rectangle(img_np, (int(box[1]), int(box[0])), (int(box[3]), int(box[2])), (0, 1, 0), 2)
            else:
                dist = class_bins[len(class_bins)//2]/10

            # cv2.imshow(f"File: {img_file}, Distance {dist}", img_np)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            _, name, ext, time_tmp = img_file.split('_', 4)
            time = time_tmp.split(".", 1)
            data_aslist.append([name+'.'+ext, time[0], dist])  #video_id,time,distance

            # plt.imshow(image.permute(1,2,0))
            # plt.show()
    df = pd.DataFrame(data_aslist, columns=['video_id','time','distance'])
    df["time"] = pd.to_numeric(df["time"])
    df.to_csv(f"sub_{name}.csv", index=False)
    # print(model)
