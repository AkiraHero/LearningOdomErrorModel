import pandas as pd
import numpy as np

if __name__ == '__main__':
    datainfofile = "/home/akira/poss-server/dataprocessing/campus0516_2/0519_0.1m_14/locmatchingdataset_mutiprocess/datainfo.csv"
    outlocalposfile = "/home/akira/poss-server/dataprocessing/DATASET/campus-data-experienced/20180519/LiDAR-Odometry/LO-Sampling/relPose-targetodom.csv"
    outlocalcovfile = "/home/akira/poss-server/dataprocessing/DATASET/campus-data-experienced/20180519/LiDAR-Odometry/LO-Sampling/tradcovmatrix-targetodom.csv"
    array = pd.read_csv(datainfofile, header=None).values
    posarr = array[:,6:]
    covarr = array[:,2:6]
    #output pos
    cnt = 1
    with open(outlocalposfile, 'w') as f:
        for i in posarr:
            s = "{},{:.6E},{:.6E},{:.6E}\n".format(cnt, i[0], i[1], i[2])
            cnt += 1
            f.write(s)

    cnt = 1
    with open(outlocalcovfile, 'w') as f:
        for line in covarr:
            # cov matrix is ordered by row
            covstr = (','.join([str(cnt)] + ['{:.6E}'] * 9) + '\n').format(line[0], line[3], 0,
                                                                           line[3], line[1], 0,
                                                                           0, 0, line[2])
            cnt += 1
            f.write(covstr)

