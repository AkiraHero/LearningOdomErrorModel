import pandas as pd
from Utils.rigidTransform import quaternion2euler
import torch
file = "/home/akira/poss-server/dataprocessing/DATASET/campus-data/LiDAR-Odometry/LO-LOAM/posecov-aloam3d.csv"
arr = pd.read_csv(file, header=None).values
#format
# time(sec), quaternion[4](xyzw), translation[3], covqq[16], covtt[9], covqt[12]
relative_posefile = "/home/akira/poss-server/dataprocessing/DATASET/campus-data/LiDAR-Odometry/LO-LOAM/relpose-RPY.csv"
relative_posefile_q = "/home/akira/poss-server/dataprocessing/DATASET/campus-data/LiDAR-Odometry/LO-LOAM/relpose-quaternion.csv"
with open(relative_posefile, 'w') as f1, open(relative_posefile_q, 'w') as f2:
    for i in arr:
        time = i[0]
        q = i[1:5]
        euler = quaternion2euler(torch.tensor(q))
        t = i[5:8]
        covqq = i[8:24]
        covtt = i[24:33]
        covqt = i[33:45]
        s_relposeRPY = (','.join(['{:.0f}'] + ['{:.6E}'] * 6) + '\n').format(time, t[0], t[1], t[2], euler[0], euler[1], euler[2])
        s_relposeQ = (','.join(['{:.0f}'] + ['{:.6E}'] * 7) + '\n').format(time, t[0], t[1], t[2], q[0], q[1], q[2], q[3])
        f1.write(s_relposeRPY)
        f2.write(s_relposeQ)