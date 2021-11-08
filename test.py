from effdet import create_model
import torch
from effdet.soft_nms import batched_soft_nms, pairwise_iou

from data import ShimpyDataModule, class_bins
from lightning import EfficientDetModel
import pandas as pd
from pytorch_lightning import Trainer
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2

chp = r"C:\Users\kir\Documents\Python\Shimpy3\lightning_logs\version_16\checkpoints\epoch=25-step=32785.ckpt"
# chp = r"C:\Users\kir\Documents\Python\Shimpy3\lightning_logs\version_44\checkpoints\epoch=15-step=19583.ckpt"
# chp = r"C:\Users\kir\Documents\Python\Shimpy3\lightning_logs\version_31\checkpoints\epoch=35-step=23399.ckpt"
# chp = r"C:\Users\kir\Documents\Python\Shimpy3\lightning_logs\version_27\checkpoints\epoch=23-step=13319.ckpt"

if __name__ == '__main__':

    image_size = [512, 512]

    dm = ShimpyDataModule(0, 2, batch_size=8, image_size=image_size, num_workers=0)
    dm.setup(random=False)

    # model = EfficientDetModel(
    #     model_architecture='tf_efficientdet_d0',
    #     num_classes=len(class_bins),
    #     img_size=512,
    # )
    # model.eval()
    # # a = model.predict(next(iter(dm.predict_dataloader())))
    # aa = next(iter(dm.predict_dataloader()))
    # a = model.model.forward(aa[0], aa[1])

    model = EfficientDetModel.load_from_checkpoint(chp)
    model.eval()
    model.wbf_iou_threshold = 0.02
    model.skip_box_thr = 0.001

    for images, targets in dm.train_dataloader():           # use this for tran data
    # for images, targets in dm.predict_dataloader():       # use this for test data
        a = model.predict(images)
        for i, image in enumerate(images):
            try:
                img_file = targets['img_file'][i]
            except:
                img_file = "no name"
            cls = a[1][i]
            img_np = image.permute(1, 2, 0).numpy().copy()
            try:
                dist_label = class_bins[int(targets['cls'][i]-1)]/10
            except:
                dist_label = "@"
            if len(cls) > 0:
                dists = [class_bins[int(c-1)] for c in cls]
                print(f"Dists:{dists} Confs:{a[2][i]}")

                sorted_conf_idx = np.argsort(a[2][i])
                strongest_box = sorted_conf_idx[-1] # np.argmax(a[2][i])

                # == average all boxes
                # dist = np.mean(dists) / 10      # 2.4509
                # == use the most confident box
                # dist = dists[strongest_box]     # 2.5239
                # == take no more than 3 best
                # if len(dists) == 1:
                #     dist = dists[strongest_box] / 10
                # elif len(dists) == 2:
                #     dist = np.mean([dists[strongest_box], dists[sorted_conf_idx[-2]]]) / 10
                # else:
                #     dist = np.mean([dists[strongest_box], dists[sorted_conf_idx[-2]], dists[sorted_conf_idx[-3]]]) / 10

                # == pick the closest predicted box to the input data box. if matches the most confident, use it. Otherwise use mean
                b_lab = torch.Tensor(targets['bbox'][i])
                b_pred = torch.Tensor(a[0][i][:])
                b_lab = torch.index_select(b_lab, 1, torch.LongTensor([1,0,3,2]))
                ious = pairwise_iou(b_pred, b_lab)
                best_box_idx = torch.argmax(ious).numpy()

                if best_box_idx == strongest_box:
                    dist = dists[best_box_idx] / 10
                else:
                    dist = np.mean(dists) / 10

                # if len(dists) == 1:
                #     dist = dists[strongest_box] / 10
                # else:
                #     dist = np.mean([dists[strongest_box], dists[sorted_conf_idx[-2]]]) / 10

                box = a[0][i][strongest_box]
                box[0], box[1] = box[1], box[0]
                box[2], box[3] = box[3], box[2]

                cv2.rectangle(img_np, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 1, 0), 2)
                target_box = targets['bbox'][i]
                if len(target_box) > 0:
                    cv2.rectangle(img_np, (int(target_box[0,0]), int(target_box[0,1])), (int(target_box[0,2]), int(target_box[0,3])), (1, 1, 0), 2)
                    print(pairwise_iou(target_box, torch.tensor([box])))
                    print(a[2][i][strongest_box])
                else:
                    print("no target box")

            else:
                dist = class_bins[len(class_bins)//2]/10

            # print(dist)
            cv2.imshow(f"{img_file},Predic:{dist},Label:{dist_label}", img_np)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


            # plt.imshow(image.permute(1,2,0))
            # plt.show()
            print('')
            # result = {
            #     'image_id': image_id,
            #     'PredictionString': format_prediction_string(boxes, scores)
            # }

            # results.append(result)

    # plt.imshow(a[0][0].permute(1,2,0))
    # plt.show()


    # trained_model = model.load_from_checkpoint(chp)
    # trainer = Trainer(gpus=[0])
    # model = EfficientDetModel.load_from_checkpoint(chp)
    # prediction = trainer.test(model,dm)
    # prediction = trainer.predict(model,dm)

    print(model)
