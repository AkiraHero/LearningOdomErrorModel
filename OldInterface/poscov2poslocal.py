import pandas as pd

if __name__ == '__main__':
    file =  "/home/akira/Project/csm/cmake-build-debug/sm/poscov0107.csv"
    outfile_pos = "/home/akira/poss-server/dataprocessing/DATASET/campus-data-largescale/LiDAR-Odometry/LO-csmICP/relPose-targetodom.csv"
    outfile_cov = "/home/akira/poss-server/dataprocessing/DATASET/campus-data-largescale/LiDAR-Odometry/LO-csmICP/tradcovmatrix-targetodom.csv"

    data = pd.read_csv(file).values

    with open(outfile_pos, 'w') as f:
        for line in data:
            s = '{:.0f},{:.6E},{:.6E},{:.6E}\n'.format(line[0], line[1], line[2], line[3])
            f.write(s)

    cnt = 1
    with open(outfile_cov, 'w') as f:
        for line in data[:, 4:]:
            s = (','.join(['{:.0f}'] + ['{:.6E}'] * 9) + '\n').format(*([cnt] + [i for i in line]))
            cnt += 1
            f.write(s)