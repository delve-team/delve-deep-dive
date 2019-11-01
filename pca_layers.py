import torch
from torch.nn import Module
from delve.torch_utils import TorchCovarianceMatrix


class LinearPCALayer(Module):

    def __init__(self, threshold: float = .99):
        super(LinearPCALayer, self).__init__()
        self.transformation_matrix = None
        self._cov = TorchCovarianceMatrix()
        self.threshold = threshold

    def _update_covariance(self, x: torch.Tensor) -> None:
        self._cov.update(x)

    def _compute_pca_matrix(self) -> torch.Tensor:
        cov_mtrx: torch.Tensor = self._cov._cov_mtx
        eig_val, eig_vec = cov_mtrx.eig(True)
        sorted_eig_val, idx = eig_val.sort(0, descending=True)
        sorted_eig_vec = eig_vec[idx]
        percentages = eig_val.cumsum(0) / eig_val.sum(0)
        eigen_space = sorted_eig_vec[percentages < self.threshold]
        self.transformation_matrix: torch.Tensor = eigen_space @ eigen_space.T

    def eval(self):
        self._compute_pca_matrix()
        return self.train(False)

    def forward(self, x):
        if self.training:
            self._update_covariance(x)
            return x
        else:
            return x @ self.transformation_matrix


class Conv2DPCALayer(Module):

    def __init__(self, threshold: float = 0.99):
        super(Conv2DPCALayer, self).__init__()
        self.convolution: torch.nn.Conv2d = None
        self.transformation_matrix = None
        self._cov = TorchCovarianceMatrix()
        self.threshold = threshold

    def _update_covariance(self, x: torch.Tensor) -> None:
        if self.convolution is None:
            self.convolution = torch.nn.Conv2d(in_channels=x.shape[1],
                                               out_channels=x.shape[1],
                                               kernel_size=1, stride=1, bias=False)
        reshaped_batch: torch.Tensor = x.permute([1, 0, 2, 3])
        reshaped_batch: torch.Tensor = reshaped_batch.flatten(1)
        reshaped_batch: torch.Tensor = reshaped_batch.permute([1, 0])
        self._cov.update(reshaped_batch)

    def _compute_pca_matrix(self):
        cov_mtrx: torch.Tensor = self._cov._cov_mtx
        eig_val, eig_vec = cov_mtrx.eig(True)
        sorted_eig_val, idx = eig_val.sort(0, descending=True)
        sorted_eig_vec = eig_vec[idx]
        percentages = eig_val.cumsum(0) / eig_val.sum(0)
        eigen_space = sorted_eig_vec[percentages < self.threshold]
        self.transformation_matrix: torch.Tensor = eigen_space @ eigen_space.T
        weight = torch.nn.Parameter(self.transformation_matrix.unsqueeze(3).unsqueeze(4))
        self.convolution.weight = weight

    def eval(self):
        self._compute_pca_matrix()
        return self.train(False)

    def forward(self, x):
        if self.training:
            self._update_covariance(x)
            return x
        else:
            return self.convolution(x)
