import torch
from math import pi as PI


# a,b shape=batchsize*1
def radian_angle_diff(a, b):
    a = torch.atan2(a.sin(), a.cos())
    b = torch.atan2(b.sin(), b.cos())
    diff = a - b
    inx = ((diff.abs() > PI) & (diff > 0.0)).nonzero()[:, 0]
    diff[inx] = diff[inx] - 2 * PI
    inx = ((diff.abs() > PI) & (diff <= 0.0)).nonzero()[:, 0]
    diff[inx] = 2 * PI + diff[inx]
    return diff


# a and b .shape = batchsize*3*1
def cal_pose_error2d(a, b):
    batch_size = a.shape[0]
    c = torch.tensor([0., 0., 0.], device=a.device).reshape(3, 1).repeat(batch_size, 1, 1)
    c[:, :2] = a[:, :2] - b[:, :2]
    c[:, 2] = radian_angle_diff(a[:, 2], b[:, 2])
    return c


# Todo check dim '2'
def predict_next_deltapose2d(newodo, lastodo, currentpos):
    transform2v = rotmat_2dcov(-lastodo[:, 2, :])
    transform2w = rotmat_2dcov(currentpos[:, 2, :])
    result = transform2w.matmul(transform2v).matmul(newodo - lastodo)
    return result


def get_velocity_model_covariance(v, omega, deltaT, Q, local_yaw, device = torch.device('cpu')):
    batchsize = v.shape[0]
    V = torch.zeros([batchsize, 3, 2], device=device)
    infinitesimal = 1e-5
    tmp = (omega > infinitesimal)
    not_infs_inx = tmp.nonzero()[:, 0]
    infs_inx = (~tmp).nonzero()[:, 0]
    if not_infs_inx.shape[0] > 0:
        omegat = omega[not_infs_inx, :] * deltaT[not_infs_inx, :]
        V[not_infs_inx, 0, 0] = (omega[not_infs_inx, :] * deltaT[not_infs_inx, :]).reshape(-1)
        V[not_infs_inx, 0, 1] = (-v[not_infs_inx, :] * omegat.sin() / (omega[not_infs_inx, :] ** 2) + v[not_infs_inx, :]
                                 * deltaT[not_infs_inx, :] * omegat.cos() / omega[not_infs_inx, :]).reshape(-1)
        V[not_infs_inx, 1, 0] = ((1 - omegat.cos()) / omega[not_infs_inx, :]).reshape(-1)
        V[not_infs_inx, 1, 1] = (-v[not_infs_inx, :] * (1 - omegat.cos()) / (omega[not_infs_inx, :] ** 2)
                                 + v[not_infs_inx, :] * deltaT[not_infs_inx, :]
                                 * omegat.sin() / omega[not_infs_inx, :]).reshape(-1)
        V[not_infs_inx, 2, 0] = 0
        V[not_infs_inx, 2, 1] = deltaT[not_infs_inx, :].reshape(-1)
    if infs_inx.shape[0] > 0:
        V[infs_inx, 0, 0] = deltaT[infs_inx, :].reshape(-1)
        V[infs_inx, 0, 1] = 0
        V[infs_inx, 1, 0] = 0
        V[infs_inx, 1, 1] = (v[infs_inx, :] * deltaT[infs_inx, :] * deltaT[infs_inx, :] / 2).reshape(-1)
        V[infs_inx, 2, 0] = 0
        V[infs_inx, 2, 1] = deltaT[infs_inx, :].reshape(-1)

    result_cov = V.matmul(Q).matmul(V.transpose(-1, -2))

    # Rotate covariance to relative coordinate
    rot = rotmat_2dcov(-local_yaw)
    result_cov = rot.matmul(result_cov).matmul(rot.transpose(-1, -2))
    return result_cov


def rotmat_2dcov(theta_batched):
    batch_size = theta_batched.shape[0]
    rot = torch.diag(torch.tensor([0., 0., 1.0], device=theta_batched.device)).unsqueeze(0).repeat(batch_size, 1, 1)
    cos_v = theta_batched.cos()
    sin_v = theta_batched.sin()
    rot[:, 0, 0] = cos_v.reshape(-1)
    rot[:, 0, 1] = -sin_v.reshape(-1)
    rot[:, 1, 0] = sin_v.reshape(-1)
    rot[:, 1, 1] = cos_v.reshape(-1)
    return rot


def rot_matrix2d(theta_batched):
    batch_size = theta_batched.shape[0]
    rot = torch.diag(torch.tensor([0., 0.], device=theta_batched.device)).unsqueeze(0).repeat(batch_size, 1, 1)
    cos_v = theta_batched.cos()
    sin_v = theta_batched.sin()
    rot[:, 0, 0] = cos_v.reshape(-1)
    rot[:, 0, 1] = -sin_v.reshape(-1)
    rot[:, 1, 0] = sin_v.reshape(-1)
    rot[:, 1, 1] = cos_v.reshape(-1)
    return rot

#pass
# pose: batchsize*3*1
def uniform_trans_matrix2d(pose):
    batch_size = pose.shape[0]
    mat = torch.diag(torch.tensor([0., 0., 1.0], device=pose.device)).unsqueeze(0).repeat(batch_size, 1, 1)
    mat[:, :2, :2] = rot_matrix2d(pose[:, 2, :])
    mat[:, :2, 2] = pose[:, :2, :].squeeze(-1)
    return mat

#pass
def get_pose2d(uni_m):
    pose = torch.tensor([0., 0., 0.], device=uni_m.device).reshape(3, 1).repeat(*uni_m.shape[:-2], 1, 1)
    pose[..., 2, :] = torch.atan2(uni_m[..., 1, 0], uni_m[..., 0, 0]).unsqueeze(-1)
    pose[..., :2, :] = uni_m[..., :2, 2].unsqueeze(-1)
    return pose

#pass
def unim_pose2d_aggregation(start_pose, relative_pose):
    uni_m = uniform_trans_matrix2d(start_pose).matmul(uniform_trans_matrix2d(relative_pose))
    return uni_m


def get_relative_pose2d(start_pose, end_pose):
    st_uni = uniform_trans_matrix2d(start_pose)
    ed_uni = uniform_trans_matrix2d(end_pose)
    return inverse_uniform_matrix2d(st_uni).matmul(ed_uni)


def get_relative_unim_pose2d(st_uni, ed_uni):
    return inverse_uniform_matrix2d(st_uni).matmul(ed_uni)


def inverse_rotm2d(rot):
    batch_size = rot.shape[0]
    mat = torch.diag(torch.tensor([0., 0.], device=rot.device)).unsqueeze(0).repeat(batch_size, 1, 1)
    mat[:, :2, :2] = rot[:, :2, :2]
    mat[:, 0, 1] = -mat[:, 0, 1]
    mat[:, 1, 0] = -mat[:, 1, 0]
    return mat


def inverse_uniform_matrix2d(uni_m):
    batch_size = uni_m.shape[0]
    mat = torch.diag(torch.tensor([0., 0., 1.0], device=uni_m.device)).unsqueeze(0).repeat(batch_size, 1, 1)
    mat[:, :2, :2] = inverse_rotm2d(uni_m[:, :2, :2])
    mat[:, :2, 2] = (-mat[:, :2, :2].matmul(uni_m[:, :2, 2].unsqueeze(-1))).squeeze(-1)
    return mat