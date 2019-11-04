import torch
from torch.nn import Module
from delve.torch_utils import TorchCovarianceMatrix


class LinearPCALayer(Module):

    def __init__(self, threshold: float = .99, keepdim=True):
        super(LinearPCALayer, self).__init__()
        self.transformation_matrix = None
        self._cov = TorchCovarianceMatrix()
        self.threshold = threshold
        self.pca_computed = False
        self.keepdim = keepdim

    def _update_covariance(self, x: torch.Tensor) -> None:
        self._cov.update(x)

    def _compute_pca_matrix(self) -> torch.Tensor:
        print('computing covariance for Linear')
        cov_mtrx: torch.Tensor = self._cov._cov_mtx
        self._cov._cov_mtx = None
        eig_val, eig_vec = cov_mtrx.symeig(True)
        sorted_eig_val, idx = eig_val.sort(descending=True)
        sorted_eig_vec = eig_vec[:, idx]
        percentages = sorted_eig_val.cumsum(0) / sorted_eig_val.sum()
        eigen_space = sorted_eig_vec[:, percentages < self.threshold]
        print(f'Saturation: {round(eigen_space.shape[1] / eig_val.shape[0], 4)}%', 'Eigenspace has shape', eigen_space.shape)
        self.transformation_matrix: torch.Tensor = eigen_space.matmul(eigen_space.t())
        self.reduced_transformation_matrix: torch.Tensor = eigen_space

    def forward(self, x):
        if self.training:
            self.pca_computed = False
            self._update_covariance(x)
            return x
        else:
            if not self.pca_computed:
                self._compute_pca_matrix()
                self.pca_computed = True
            if self.keepdim:
                return x @ self.transformation_matrix.t()
            else:
                return x @ self.reduced_transformation_matrix


class Conv2DPCALayer(Module):

    def __init__(self, threshold: float = 0.99):
        super(Conv2DPCALayer, self).__init__()
        print('Added Conv2D PCA Layer')
        self.convolution: torch.nn.Conv2d = None
        self.transformation_matrix: torch.Tensor = None
        self._cov = TorchCovarianceMatrix()
        self.threshold = threshold
        self.pca_computed = False

    def _update_covariance(self, reshaped_batch: torch.Tensor) -> None:
        self._cov.update(reshaped_batch)


    def _compute_pca_matrix(self):
        print('computing covariance for Conv2D')
        cov_mtrx: torch.Tensor = self._cov._cov_mtx
        self._cov._cov_mtx = None
        eig_val, eig_vec = cov_mtrx.symeig(True)
        sorted_eig_val, idx = eig_val.sort(descending=True)
        sorted_eig_vec = eig_vec[:, idx]
        percentages = sorted_eig_val.cumsum(0) / sorted_eig_val.sum()
        eigen_space = sorted_eig_vec[:, percentages < self.threshold]
        print(f'Saturation: {round(eigen_space.shape[1] / eig_val.shape[0], 4)}%', 'Eigenspace has shape', eigen_space.shape)
        self.transformation_matrix: torch.Tensor = eigen_space.matmul(eigen_space.t())
        weight = torch.nn.Parameter(self.transformation_matrix.unsqueeze(2).unsqueeze(3))
        self.convolution.weight = weight

    def forward(self, x):
        if self.training:
            self.pca_computed = False
            if self.convolution is None:
                print('initializing convolution')
                self.convolution = torch.nn.Conv2d(in_channels=x.shape[1],
                                                   out_channels=x.shape[1],
                                                   kernel_size=1, stride=1, bias=False)
                self.convolution.to('cuda:0')
            swapped: torch.Tensor = x.permute([1, 0, 2, 3])
            flattened: torch.Tensor = swapped.flatten(1)
            reshaped_batch: torch.Tensor = flattened.permute([1, 0])
            self._update_covariance(reshaped_batch)
            return x
        else:
            if not self.pca_computed:
                self._compute_pca_matrix()
            self.pca_computed = True
            return self.convolution(x)
