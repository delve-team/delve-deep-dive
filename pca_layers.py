import torch
from torch.nn import Module


def change_all_pca_layer_thresholds(threshold: float, network: Module, verbose: bool = False):
    for module in network.modules():
        if isinstance(module, Conv2DPCALayer) or isinstance(module, LinearPCALayer):
            module.threshold = threshold
            if verbose:
                print(f'Changed threshold for layer {module} to {threshold}')


class LinearPCALayer(Module):

    def __init__(self, in_features: int, threshold: float = .99, keepdim: bool = True, verbose: bool = False):
        super(LinearPCALayer, self).__init__()
        self.register_buffer('eigenvalues', torch.zeros(in_features))
        self.register_buffer('eigenvectors', torch.zeros((in_features, in_features)))
        self.register_buffer('_threshold', torch.Tensor([threshold]))
        self.register_buffer('autorcorrelation_matrix', torch.zeros((in_features, in_features)))
        self.register_buffer('seen_samples', torch.zeros(1))
        self.keepdim: bool = keepdim
        self.verbose: bool = verbose
        self.pca_computed: bool = True

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, threshold: float) -> None:
        self._threshold.data = torch.Tensor([threshold]).to(self.threshold.device)
        self._compute_pca_matrix()

    def _update_autorcorrelation(self, x: torch.Tensor) -> None:
        self.autorcorrelation_matrix.data += torch.matmul(x.transpose(0, 1), x)
        self.seen_samples.data += x.shape[0]

    def _compute_autorcorrelation(self) -> torch.Tensor:
        tlen = self.seen_samples
        cov_mtx = self.autorcorrelation_matrix
        cov_mtx /= tlen - 1
        return cov_mtx

    def _compute_eigenspace(self):
        self.eigenvalues.data, self.eigenvectors.data = self._compute_autorcorrelation().symeig(True)
        self.eigenvalues.data, idx = self.eigenvalues.sort(descending=True)
        # correct numerical error, matrix must be positivly semi-definitie
        self.eigenvalues[self.eigenvalues < 0] = 0
        self.eigenvectors.data = self.eigenvectors[:, idx]

    def _reset_autorcorrelation(self):
        self.autorcorrelation_matrix.data = torch.zeros(self.autorcorrelation_matrix.shape).to(self.autorcorrelation_matrix.device)
        self.seen_samples += torch.zeros(self.seen_samples.shape).to(self.autorcorrelation_matrix.device)

    def _compute_pca_matrix(self):
        if self.verbose:
            print('computing autorcorrelation for Linear')
        percentages = self.eigenvalues.cumsum(0) / self.eigenvalues.sum()
        eigen_space = self.eigenvectors[:, percentages < self.threshold]
        if self.verbose:
            print(f'Saturation: {round(eigen_space.shape[1] / self.eigenvalues.shape[0], 4)}%', 'Eigenspace has shape', eigen_space.shape)
        self.transformation_matrix: torch.Tensor = eigen_space.matmul(eigen_space.t())
        self.reduced_transformation_matrix: torch.Tensor = eigen_space

    def forward(self, x):
        if self.training:
            self.pca_computed = False
            self._update_autorcorrelation(x)
            return x
        else:
            if not self.pca_computed:
                self._compute_autorcorrelation()
                self._compute_eigenspace()
                self._compute_pca_matrix()
                self.pca_computed = True
                self._reset_autorcorrelation()
            if self.keepdim:
                return x @ self.transformation_matrix.t()
            else:
                return x @ self.reduced_transformation_matrix


class Conv2DPCALayer(LinearPCALayer):

    def __init__(self, in_filters, threshold: float = 0.99, verbose: bool = True):
        super(Conv2DPCALayer, self).__init__(in_features=in_filters, threshold=threshold, keepdim=True, verbose=verbose)
        if verbose:
            print('Added Conv2D PCA Layer')
        self.convolution = torch.nn.Conv2d(in_channels=in_filters,
                                           out_channels=in_filters,
                                           kernel_size=1, stride=1, bias=False)

    def _compute_pca_matrix(self):
        if self.verbose:
            print('computing autorcorrelation for Conv2D')
        super()._compute_pca_matrix()
        # unsequeeze the matrix into 1x1xDxD in order to make it behave like a 1x1 convolution
        weight = torch.nn.Parameter(self.transformation_matrix.unsqueeze(2).unsqueeze(3))
        self.convolution.weight = weight

    def forward(self, x):
        if self.training:
            self.pca_computed = False
            swapped: torch.Tensor = x.permute([1, 0, 2, 3])
            flattened: torch.Tensor = swapped.flatten(1)
            reshaped_batch: torch.Tensor = flattened.permute([1, 0])
            self._update_autorcorrelation(reshaped_batch)
            return x
        else:
            if not self.pca_computed:
                self._compute_autorcorrelation()
                self._compute_eigenspace()
                self._compute_pca_matrix()
                self._reset_autorcorrelation()
                self.pca_computed = True
            return self.convolution(x)
