import torch
from effdet.soft_nms import pairwise_iou

from data import ShimpyDataModule, class_bins
from lightning import EfficientDetModel
import pandas as pd
from pytorch_lightning import Trainer
import matplotlib
import numpy as np
import cv2

from tqdm import tqdm
import colorama         # fix tqdm colour bug
colorama.deinit()
colorama.init(strip=False)

import warnings
warnings.filterwarnings("ignore")

chp = r"C:\Users\kir\Documents\Python\Shimpy3\lightning_logs\version_47\checkpoints\epoch=27-step=34271.ckpt"
output_name = "m5-highres-fancy-box-lowthres"

# chp = r"C:\Users\kir\Documents\Python\Shimpy3\lightning_logs\version_44\checkpoints\epoch=15-step=19583.ckpt"
# output_name = "m4-new-crop"

# chp = r"C:\Users\kir\Documents\Python\Shimpy3\lightning_logs\version_32\checkpoints\epoch=27-step=34271.ckpt"
# output_name = "m3-onefold-bb-0.05"

# chp = r"C:\Users\kir\Documents\Python\Shimpy3\lightning_logs\version_29\checkpoints\epoch=35-step=22499.ckpt"
# chp = r"C:\Users\kir\Documents\Python\Shimpy3\lightning_logs\version_30\checkpoints\epoch=35-step=22427.ckpt"
# chp = r"C:\Users\kir\Documents\Python\Shimpy3\lightning_logs\version_31\checkpoints\epoch=35-step=23399.ckpt"
# output_name = 'm2-fold2-thr0.3'

# chp = r"C:\Users\kir\Documents\Python\Shimpy3\lightning_logs\version_27\checkpoints\epoch=23-step=13319.ckpt"
# name = 'm1-fold0'
# chp = r"C:\Users\kir\Documents\Python\Shimpy3\lightning_logs\version_28\checkpoints\epoch=29-step=16949.ckpt"
# name = 'm1-fold1'

if __name__ == '__main__':
    device = 'cuda'
    image_size = [512, 512]

    dm = ShimpyDataModule(0, 2, batch_size=10, image_size=image_size, num_workers=0)
    dm.setup()

    model = EfficientDetModel.load_from_checkpoint(chp)
    model.eval()
    model.wbf_iou_threshold = 0.2       # 0.3 - хуже. 0.02 - тоже хуже
    model.skip_box_thr = 0.02

    model.to(device)
    data_aslist = []

    with tqdm(dm.predict_dataloader(), unit="batch", mininterval=0.2, colour='green') as tepoch:
        for images, targets in tepoch:
            a = model.predict(images.to(device))
            for i, image in enumerate(images):
                img_file = targets['img_file'][i]
                cls = a[1][i]
                if len(cls) > 0:
                    dists = [class_bins[int(c-1)] for c in cls]

                    # == pick the closest predicted box to the input data box. if matches the most confident, use it. Otherwise use mean
                    # strongest_box = np.argmax(a[2][i])
                    # b_lab = torch.Tensor(targets['bbox'][i])
                    # b_pred = torch.Tensor(a[0][i][:])
                    # b_lab = torch.index_select(b_lab, 1, torch.LongTensor([1, 0, 3, 2]))
                    # ious = pairwise_iou(b_pred, b_lab)
                    # best_box_idx = torch.argmax(ious).numpy()
                    # if best_box_idx == strongest_box:
                    #     dist = dists[best_box_idx] / 10
                    # else:
                    #     dist = np.mean(dists) / 10

                    # == it seems that this simple way works better?
                    dist = np.mean(dists) / 10

                    # box = a[0][i][strongest_box]
                    # box[0], box[1] = box[1], box[0]
                    # box[2], box[3] = box[3], box[2]
                    # target_box = targets['bbox'][i]

                    # iou = pairwise_iou(target_box, torch.tensor([box]))
                    # confidence is not a good criteria!
                    # but if detected box mistmatch the label box, we don't trust the distance (class)
                    # so, if UOI < 0.15 (or even 0.2?), we use average distance
                    # no - No joy!!!
                    # if iou < 0.05:
                    #     dist = class_bins[len(class_bins) // 2] / 10

                    # img_np = image.permute(1, 2, 0).numpy().copy()
                    # cv2.rectangle(img_np, (int(box[1]), int(box[0])), (int(box[3]), int(box[2])), (0, 1, 0), 2)
                    # cv2.rectangle(img_np, (int(target_box[0, 0]), int(target_box[0, 1])),
                    #               (int(target_box[0, 2]), int(target_box[0, 3])), (1, 1, 0), 2)
                else:
                    dist = class_bins[len(class_bins)//2]/10

                # do we also need to decimate the output distance?

                # cv2.imshow(f"File: {img_file}, Distance {dist}", img_np)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                _, name, ext, time_tmp = img_file.split('_', 4)
                time = time_tmp.split(".", 1)
                data_aslist.append([name+'.'+ext, time[0], dist])  #video_id,time,distance

    df = pd.DataFrame(data_aslist, columns=['video_id','time','distance'])
    df["time"] = pd.to_numeric(df["time"])
    df.to_csv(f"sub_{output_name}.csv", index=False)
    # print(model)
