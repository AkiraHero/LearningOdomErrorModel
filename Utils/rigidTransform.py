import torch
# import quaternion


"""Operation on rigid body Rotation
"""


def rotation_matrix2euler(m):
    if isinstance(m, torch.Tensor):
        assert m.shape == torch.Size([3, 3])
        sy = torch.sqrt(m[0][0] * m[0][0] + m[1][0] * m[1][0])
        singular = sy < 1e-6
        global x, y, z
        if not singular:
            x = torch.atan2(m[2][1], m[2][2])
            y = torch.atan2(-m[2][0], sy)
            z = torch.atan2(m[1][0], m[0][0])
        else:
            x = torch.atan2(-m[1][2], m[1][1])
            y = torch.atan2(-m[2][0], sy)
            z = 0
        return torch.cat((x.reshape(1), y.reshape(1), z.reshape(1)))


def rotation_matrix2quaternion(m):
    if isinstance(m, torch.Tensor):
        assert m.shape == torch.Size([3, 3])
        tr = m.trace()
        global qx,qy,qz,qw
        if tr > 0:
            S = torch.sqrt(tr + 1.0) * 2 # S=4 * qw
            qw = 0.25 * S
            qx = (m[2][1] - m[1][2]) / S
            qy = (m[0][2] - m[2][0]) / S
            qz = (m[1][0] - m[0][1]) / S
        elif (m[0][0] > m[1][1]) and (m[0][0] > m[2][2]):
            S = torch.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2 # S=4 * qx
            qw = (m[2][1] - m[1][2]) / S
            qx = 0.25 * S
            qy = (m[0][1] + m[1][0]) / S
            qz = (m[0][2] + m[2][0]) / S
        elif (m[1][1] > m[2][2]):
            S = torch.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2 # S=4 * qy
            qw = (m[0][2] - m[2][0]) / S
            qx = (m[0][1] + m[1][0]) / S
            qy = 0.25 * S
            qz = (m[1][2] + m[2][1]) / S
        else:
            S = torch.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2 # S=4 * qz
            qw = (m[1][0] - m[0][1]) / S
            qx = (m[0][2] + m[2][0]) / S
            qy = (m[1][2] + m[2][1]) / S
            qz = 0.25 * S
        return torch.cat((qx.reshape(1),qy.reshape(1),qz.reshape(1),qw.reshape(1)))


def euler2rotation_matrix(e):
    if isinstance(e, torch.Tensor):
        # assert e.size() == torch.Size([3])
        roll = e[..., 0, 0]
        pitch = e[..., 1, 0]
        yaw = e[..., 2, 0]
        matrix = torch.zeros(*e.shape[:-2], 3, 3, device = e.device)
        ca = torch.cos(yaw)
        cb = torch.cos(pitch)
        cr = torch.cos(roll)
        sa = torch.sin(yaw)
        sb = torch.sin(pitch)
        sr = torch.sin(roll)
        matrix[..., 0, 0] = ca * cb
        matrix[..., 0, 1] = ca * sb * sr - sa * cr
        matrix[..., 0, 2] = ca * sb * cr + sa * sr
        matrix[..., 1, 0] = sa * cb
        matrix[..., 1, 1] = sa * sb * sr + ca * cr
        matrix[..., 1, 2] = sa * sb * cr - ca * sr
        matrix[..., 2, 0] = -sb
        matrix[..., 2, 1] = cb * sr
        matrix[..., 2, 2] = cb * cr
        return matrix


def quaternion2rotation_matrix(q):
    if isinstance(q, torch.Tensor):
        assert q.size() == torch.Size([4])
        matrix = torch.zeros(3, 3, device = q.device)
        x_ = q[0]
        y_ = q[1]
        z_ = q[2]
        w_ = q[3]
        matrix[0][0] = (1 - 2 * y_ * y_ - 2 * z_ * z_)
        matrix[0][1] = 2 * (x_ * y_ - z_ * w_)
        matrix[0][2] = 2 * (x_ * z_ + y_ * w_)
        matrix[1][0] = 2 * (x_ * y_ + z_ * w_)
        matrix[1][1] = (1 - 2 * x_ * x_ - 2 * z_ * z_)
        matrix[1][2] = 2 * (y_ * z_ - x_ * w_)
        matrix[2][0] = 2 * (x_ * z_ - y_ * w_)
        matrix[2][1] = 2 * (y_ * z_ + x_ * w_)
        matrix[2][2] = (1 - 2 * y_ * y_ - 2 * x_ * x_)
        return matrix


def euler2quaternion(e):
    if isinstance(e, torch.Tensor):
        assert e.size() == torch.Size([3])
        roll = e[0]
        pitch = e[1]
        yaw = e[2]
        c_r = torch.cos(roll / 2)
        c_p = torch.cos(pitch / 2)
        c_y = torch.cos(yaw / 2)
        s_r = torch.sin(roll / 2)
        s_p = torch.sin(pitch / 2)
        s_y = torch.sin(yaw / 2)
        w = c_r * c_p * c_y + s_r * s_p * s_y
        x = s_r * c_p * c_y - c_r * s_p * s_y
        y = c_r * s_p * c_y + s_r * c_p * s_y
        z = c_r * c_p * s_y - s_r * s_p * c_y
        # Quaternion equation: xi + yj + zk + w
        return torch.cat((x.reshape(1),y.reshape(1),z.reshape(1),w.reshape(1)))


def quaternion2euler(q):#
    if isinstance(q, torch.Tensor):
        assert q.size() == torch.Size([4])
        roll = torch.atan2(2 * (q[3] * q[0] + q[1] * q[2]), 1 - 2 * (q[0] * q[0] + q[1] * q[1]))
        pitch = torch.asin(2 * (q[3] * q[1] - q[0] * q[2]))
        yaw = torch.atan2(2 * (q[3] * q[2] + q[1] * q[0]), 1 - 2 * (q[2] * q[2] + q[1] * q[1]))
        return torch.cat((roll.reshape(1), pitch.reshape(1), yaw.reshape(1)))


"""Operation on Uniform TransMatrix
"""
def xyzrpy2uniform_matrix(t):
    assert t.shape[-2] == 6 and t.shape[-1] == 1
    unim = torch.zeros(*t.shape[:-2], 4, 4, device=t.device)
    unim[..., :3, :3] = euler2rotation_matrix(t[..., 3:6, :])
    unim[..., :3, 3] = t[..., :3, 0]
    unim[..., 3, 3] = 1
    return unim

def get_position_from_uniform_matrix(m):
    if isinstance(m, torch.Tensor):
        assert m.shape == torch.Size([4, 4])
        return m[0:3,3]


def get_rotation_matrix_in_uniform_matrix(m):
    if isinstance(m, torch.Tensor):
        assert m.shape == torch.Size([4, 4])
        return m[0:3, 0:3]


def get_relative_transition(new_uniformM, last_uniformM):
    assert isinstance(new_uniformM, torch.Tensor) and isinstance(last_uniformM, torch.Tensor)
    assert new_uniformM.size() == torch.Size([4, 4]) and last_uniformM.size() == torch.Size([4, 4])
    # Equation: last * R = new
    R = last_uniformM.inverse().matmul(new_uniformM)
    return R


def get_global_pose(relative_uniformM, base_uniformM):
    assert isinstance(relative_uniformM, torch.Tensor) and isinstance(base_uniformM, torch.Tensor)
    assert relative_uniformM.size() == torch.Size([4, 4]) and base_uniformM.size() == torch.Size([4, 4])
    return base_uniformM.matmul(relative_uniformM)


def relative2relative_uniformM(ini_relativeM, base_diff):
    assert isinstance(ini_relativeM, torch.Tensor) and isinstance(base_diff, torch.Tensor)
    assert ini_relativeM.size() == torch.Size([4, 4]) and base_diff.size() == torch.Size([4, 4])
    return base_diff.inverse().matmul(ini_relativeM).matmul(base_diff)


# matshape N*...*matrixshape
# N: dim for multiplication
def factor_matmul(mat):
    muldimsize = mat.shape[0]
    res = muldimsize % 2
    factor_size = muldimsize - res
    product = mat[0:factor_size:2].matmul(mat[1:factor_size:2])
    if res:
        product = torch.cat([product, mat[-1].unsqueeze(0)], dim=0)
    return product
from Utils.debugUtils import measure_time
# @measure_time()
def fast_matmul(mat):
    while mat.shape[0] > 1:
        mat = factor_matmul(mat)
    return mat.squeeze(0)

# cov1: cov mat of relative pose1(x1,y1,theta1) to pose0(0,0,0)
# cov2: cov mat of relative pose2(x2,y2,theta2) to pose0(0,0,0)
# to get cov of new pose1 (+) pose2 relative to pose0(0,0,0)
def propagate_cov2d(pos1, pos2, cov1, cov2):
    batchsize = pos1.shape[0]
    x2 = pos2[..., 0, 0]
    y2 = pos2[..., 1, 0]
    theta1 = pos1[..., 2, 0]
    G = torch.eye(3, device=pos1.device).repeat(batchsize, 1, 1)
    G[..., 0, 2] = -x2 * torch.sin(theta1) - y2 * torch.cos(theta1)
    G[..., 1, 2] = x2 * torch.cos(theta1) - y2 * torch.sin(theta1)
    V = torch.eye(3, device=pos1.device).repeat(batchsize, 1, 1)
    V[..., 0, 0] = torch.cos(theta1)
    V[..., 0, 1] = -torch.sin(theta1)
    V[..., 1, 0] = torch.sin(theta1)
    V[..., 1, 1] = torch.cos(theta1)
    return G.matmul(cov1).matmul(G.transpose(-1, -2)) + V.matmul(cov2).matmul(V.transpose(-1, -2))



