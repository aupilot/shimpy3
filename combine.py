import pandas as pd
import numpy as np

pred_list = [
    'sub_m2-fold0.csv',
    'sub_m2-fold1.csv',
    'sub_m2-fold2.csv',
    ]

dff=[]
for file_name in pred_list:
    dff.append(pd.read_csv(file_name))

n_combine = len(pred_list)

new_submit = []
for index, row in dff[0].iterrows():
    dist = row['distance']
    for df in dff[2:]:
        tmp = df.loc[(df['video_id'] == row['video_id']) & (df['time'] == row['time'])]
        dist = dist + tmp['distance'].item()
    dist = dist / n_combine
    new_submit.append([row['video_id'], row['time'], dist])

new_df = pd.DataFrame(new_submit, columns=['video_id','time','distance'])
new_df.to_csv("sub_combined.csv", index=False)


