import pandas as pd
import torch
from Utils.rigidTransform import xyzrpy2uniform_matrix, get_position_from_uniform_matrix
import matplotlib.pyplot as plt
import numpy as np
if __name__ == "__main__":
    posefile = "/home/akira/poss-server/dataprocessing/DATASET/campus-data/LiDAR-Odometry/LO-LOAM/relpose-RPY.csv"
    arr = pd.read_csv(posefile, header=None).values
    xyzrpy = torch.tensor(arr[:, 1:]).unsqueeze(-1)
    unim = xyzrpy2uniform_matrix(xyzrpy)
    res = torch.eye(4)
    poslist = []
    for i in unim:
        res = res.matmul(i)
        pos = get_position_from_uniform_matrix(res)
        poslist.append(pos.cpu().numpy())
    poslist = np.array(poslist)
    plt.plot(poslist[:, 0], poslist[:, 1])
    plt.show()
    pass
