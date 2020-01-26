import torch
from Utils.fusionUtils import rotmat_2dcov

class RelativePositionFuser:
    def __init__(self):
        self._device = torch.device('cpu')
        self._initial_information_matrix = torch.diag(torch.tensor([100000, 100000, 100000], device=self.device))
        self._information_matrix = None
        self.batch_size = None

    @property
    def initial_information_matrix(self):
        return self._initial_information_matrix

    @initial_information_matrix.setter
    def initial_information_matrix(self, info):
        self._initial_information_matrix = info


    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, dev):
        self._device = dev

    def set_batch_size(self, batchsize):
        self.batch_size = batchsize

    def reinit(self):
        self._information_matrix = self._initial_information_matrix.clone().detach().unsqueeze(0).repeat(self.batch_size, 1, 1)

    #TODO: figure out whether it is necessary to reinit information matrix
    def step_fusion(self, result1, covariance_matrix1, result2, information_matrix2, fuse_scale):
        if self._information_matrix is None:
            self.reinit()
        real_batch_size = covariance_matrix1.shape[0]
        omega_ = (self._information_matrix[:real_batch_size, :, :].inverse() + covariance_matrix1).inverse()
        kese_ = torch.matmul(omega_, result1)
        if fuse_scale:
            omega = omega_ + information_matrix2
            kese = kese_ + torch.matmul(information_matrix2, result2)
        else:
            H = self.getHofVO(result1)
            omega = omega_ + H.transpose(-1, -2).matmul(information_matrix2).matmul(H)
            kese = kese_ + H.transpose(-1, -2).matmul(information_matrix2)\
                .matmul(result2 - self.getNoScaleRelPose(result1) + H.matmul(result1))
        new_result = omega.inverse().matmul(kese)
        rot = rotmat_2dcov(-new_result[:, 2, :])

        local_omega = rot.matmul(omega).matmul(rot.transpose(-1, -2))

        self._information_matrix = local_omega
        return new_result, local_omega, omega

    @staticmethod
    def getHofVO(pose):
        x = pose[..., 0, 0]
        y = pose[..., 1, 0]
        t = pose[..., 2, 0]

        H = torch.zeros((*pose.shape[:-2], 3, 3), device=pose.device)
        squaresum = x ** 2 + y ** 2
        infinitesimal = 1e-5
        tmp = (squaresum > infinitesimal)
        not_infs_inx = tmp.nonzero()[:, 0]
        infs_inx = (~tmp).nonzero()[:, 0]
        if not_infs_inx.shape[0] > 0:
            H[not_infs_inx, 0, 0] = 1 / ((squaresum[not_infs_inx]) ** (1 / 2)) - x[not_infs_inx] ** 2 / ((squaresum[not_infs_inx]) ** (3 / 2))
            H[not_infs_inx, 0, 1] = -(x[not_infs_inx] * y[not_infs_inx]) / ((squaresum[not_infs_inx]) ** (3 / 2))
            H[not_infs_inx, 0, 2] = 0

            H[not_infs_inx, 1, 0] = -(x[not_infs_inx] * y[not_infs_inx]) / ((squaresum[not_infs_inx]) ** (3 / 2))
            H[not_infs_inx, 1, 1] = 1 / ((squaresum[not_infs_inx]) ** (1 / 2)) - y[not_infs_inx] ** 2 / ((squaresum[not_infs_inx]) ** (3 / 2))
            H[not_infs_inx, 1, 2] = 0

            H[not_infs_inx, 2, 0] = 0
            H[not_infs_inx, 2, 1] = 0
            H[not_infs_inx, 2, 2] = 1
        if infs_inx.shape[0] > 0:
            H[infs_inx, 0, 0] = 1
            H[infs_inx, 0, 1] = 0
            H[infs_inx, 0, 2] = 0

            H[infs_inx, 1, 0] = 0
            H[infs_inx, 1, 1] = 1
            H[infs_inx, 1, 2] = 0

            H[infs_inx, 2, 0] = 0
            H[infs_inx, 2, 1] = 0
            H[infs_inx, 2, 2] = 1
        return H

    @staticmethod
    def getNoScaleRelPose(pose):
        squaresum = pose[..., 0, 0] ** 2 + pose[..., 1, 0] ** 2
        infinitesimal = 1e-5
        tmp = (squaresum > infinitesimal)
        not_infs_inx = tmp.nonzero()[:, 0]
        infs_inx = (~tmp).nonzero()[:, 0]

        lmat = torch.zeros((*pose.shape[:-2], 3, 3), device=pose.device)
        if not_infs_inx.shape[0] > 0:
            lmat[not_infs_inx, 0, 0] = lmat[not_infs_inx, 1, 1] = 1 / squaresum[not_infs_inx].sqrt()
        if infs_inx.shape[0] > 0:
            lmat[infs_inx, 0, 0] = lmat[not_infs_inx, 1, 1] = 1
        lmat[..., 2, 2] = 1
        return lmat.matmul(pose)



