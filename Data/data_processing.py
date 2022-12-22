# json reader

import json, pdb
import numpy as np
import torch
import pandas as pd
from pandas import DataFrame
import datetime

time =datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")


result = 'match_result-2022-12-22-17-13.json'
with open(result, 'r') as f:
    content = f.read()
retrieval_data = json.loads(content)
q_id = retrieval_data["q_img"]
re_id = retrieval_data["retrieval_index"]
unc = retrieval_data["similarity"]

pose_path='poses.csv'
poses = torch.from_numpy(np.loadtxt(pose_path, delimiter=',')) #(15000,7)

save_path = 'EST-'+time+'.csv'
columns=['q_id', 're_id','gt-lat','gt-log','gt-heg','gt-yaw', 'gt-pitch', 'gt-row','lat','log','heg','yaw', 'pitch', 'row']
df = pd.DataFrame(columns=columns)

for i in range(len(re_id)):  
     
    data = [[q_id[i][14:-4]]+[re_id[i]]+poses[int(q_id[i][14:-4])][1:].numpy().tolist()+poses[re_id[i]][1:].numpy().tolist()]
    wgs = pd.DataFrame(data=data, columns=columns)
    df=pd.concat([df, wgs], ignore_index=True)

DataFrame(df).to_csv(save_path, index=False, header=True)
print('Process Done. \nResult saved at: {}'.format(save_path))