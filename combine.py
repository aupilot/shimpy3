import pandas as pd
import numpy as np

pred_list = [
    'sub_m2-fold0.csv',
    'sub_m2-fold1.csv',
    'sub_m2-fold2.csv',
    # 'sub_m4-new-crop.csv',
    ]

# labels['distance'].unique()
allowed = np.array([ 3. ,  4. ,  1.5, 17. , 16. ,  5. , 14. ,  4.5, 10. , 11. ,  7. ,
        6. ,  2. ,  9. ,  1. , 13. , 12. , 15. ,  8. ,  2.5,  0.5,  7.5,
        6.5, 20. ,  5.5,  3.5, 25. , 13.5, 18. , 21. , 22. , 19. , 23. ,
       24. ])
# allowed = np.array(class_bins)/10.


def decimate(number):
    # improves 3.4380 to 3.4347
    return allowed[np.abs(allowed-number).argmin()]


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
    # if decimate
    dist = decimate(dist)
    new_submit.append([row['video_id'], row['time'], dist])

new_df = pd.DataFrame(new_submit, columns=['video_id','time','distance'])
new_df.to_csv("sub_combined.csv", index=False)


