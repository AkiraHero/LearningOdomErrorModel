import pandas as pd
from Training.ErrorModelLearner import ErrorModelLearner
import torch
from Utils.fusionUtils import get_velocity_model_covariance


def process_odominfo(data, time_step):
    dev = data['odomresult'].device
    batchsize = data['odomresult'].shape[0]
    # Get the corresponding self-odometry result, such as wheel encoder/IMU suite, shape=batch_size*3*1
    odom_result = data['odomresult'][:, time_step, 1:].unsqueeze(-1)

    # Get other information for the error model prediction of self-odometry above, shape=batch_size*8*1
    odom_information = data['odominfo'][:, time_step, 1:].unsqueeze(-1)

    # Parse self-odometry information to independent variables
    velocity = odom_information[:, 0, :]
    omega_velocity = odom_information[:, 1, :]
    sigma2_V = odom_information[:, 2, :]
    sigma2_W = odom_information[:, 3, :]
    delta_time = odom_information[:, 4, :]

    # Refer to the book "Probabilistic Robotics"
    odom_VW_matrix = torch.zeros(batchsize, 2, 2).to(dev)
    odom_VW_matrix[:, 0, 0] = sigma2_V.reshape(-1)
    odom_VW_matrix[:, 1, 1] = sigma2_W.reshape(-1)

    odom_covariance_velocity_model = get_velocity_model_covariance(velocity, omega_velocity, delta_time,
                                                                   odom_VW_matrix, odom_result[:, 2],
                                                                   device=dev)
    return odom_result, odom_covariance_velocity_model




if __name__ == '__main__':
    infofile = "/home/akira/Project/gettraj/cmake-build-debug/svo/infosvo.csv"
    infoarr = pd.read_csv(infofile, header=None).values
    rel_odomfile = "/home/akira/Project/gettraj/cmake-build-debug/svo/noisylocalsvo.csv"
    rel_odomarr = pd.read_csv(rel_odomfile, header=None).values
    outlocalcovfile = "/home/akira/server2/dataprocessing/DATASET/campus-data-largescale/Visual-Odometry/VO-PLSVO/tradcovmatrix-EncoderDR.csv"
    assert infoarr.shape[0] == rel_odomarr.shape[0]
    batchsize = infoarr.shape[0]
    data = {}
    # size: batchsize * timestep * 3 * 1
    data['odomresult'] = torch.tensor(rel_odomarr).unsqueeze(-2).float()
    # size: batchsize * timestep * 8 * 1
    data['odominfo'] = torch.tensor(infoarr).unsqueeze(-2).float()
    odom_result, odom_covariance_velocity_model = process_odominfo(data, 0)
    index = data['odominfo'][:, 0, 0]
    with open(outlocalcovfile, 'w') as f:
        for inx, mat in zip(index, odom_covariance_velocity_model):
            s = (','.join(['{:.0f}'] + ['{:.6E}'] * 9) + '\n').format(inx.item(),
                                                                 mat[0][0].item(), mat[0][1].item(), mat[0][2].item(),
                                                                 mat[1][0].item(), mat[1][1].item(), mat[1][2].item(),
                                                                 mat[2][0].item(), mat[2][1].item(), mat[2][2].item(),
                                                                 )
            f.write(s)