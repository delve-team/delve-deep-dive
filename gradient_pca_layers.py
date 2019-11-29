import torch
from torch.nn import Module
from torch.autograd import Function
from torch.nn.functional import conv2d


def change_all_pca_layer_thresholds(threshold: float, network: Module, verbose: bool = False):
    for module in network.modules():
        if isinstance(module, Conv2DPCALayer) or isinstance(module, LinearPCALayer):
            module.threshold = threshold
            if verbose:
                print(f'Changed threshold for layer {module} to {threshold}')


class LinearPCALayerFunction(Function):

    @staticmethod
    def forward(ctx, x, transformation_matrix):
        ctx.transformation_matrix = transformation_matrix
        if transformation_matrix is not None and False:
            transformation_matrix = transformation_matrix.to(x.device)
            x = x @ transformation_matrix
        return x

    @staticmethod
    def backward(ctx, grad_output):
        transformation_matrix = ctx.transformation_matrix
        if transformation_matrix is not None:
            #print('Linear computing gradient projection')
            return grad_output @ transformation_matrix, None
        return grad_output, None


class Conv2DPCALayerFunction(Function):

    @staticmethod
    def forward(ctx, x, trans_conv=None):
        ctx.trans_conv = trans_conv
        if trans_conv is not None and False:
            x = trans_conv(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        trans_conv = ctx.trans_conv
        if trans_conv is not None:
           # print('Conv2D Gradient Projection')
            return trans_conv(grad_output), None
        return grad_output, None


class LinearPCALayer(Module):

    def __init__(self, in_features: int, threshold: float = .99, keepdim: bool = True, verbose: bool = False, gradient_epoch_start: int = 1, centering: bool = False, boosted: bool = False):
        super(LinearPCALayer, self).__init__()
        self.register_buffer('eigenvalues', torch.zeros(in_features))
        self.register_buffer('eigenvectors', torch.zeros((in_features, in_features)))
        self.register_buffer('_threshold', torch.Tensor([threshold]))
        self.register_buffer('autorcorrelation_matrix', torch.zeros((in_features, in_features)))
        self.register_buffer('seen_samples', torch.zeros(1))
        self.register_buffer('running_sum', torch.zeros(in_features))
        self.register_buffer('mean', torch.zeros(in_features))
        self.keepdim: bool = keepdim
        self.verbose: bool = verbose
        self.pca_computed: bool = True
        self.gradient_epoch = gradient_epoch_start
        self.epoch = 0
        self.centering = centering
        self.boosted = boosted

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, threshold: float) -> None:
        self._threshold.data = torch.Tensor([threshold]).to(self.threshold.device)
        self._compute_pca_matrix()

    def _update_autorcorrelation(self, x: torch.Tensor) -> None:
        self.autorcorrelation_matrix.data += torch.matmul(x.transpose(0, 1), x)
        self.running_sum += x.sum(dim=0)
        self.seen_samples.data += x.shape[0]

    def _compute_autorcorrelation(self) -> torch.Tensor:
        tlen = self.seen_samples
        cov_mtx = self.autorcorrelation_matrix
        cov_mtx /= tlen - 1
        avg = self.running_sum / tlen
        if self.centering:
            avg_mtx = torch.ger(avg, avg)
            cov_mtx = cov_mtx - avg_mtx
        self.mean.data = avg
        return cov_mtx

    def _compute_eigenspace(self):
        self.eigenvalues.data, self.eigenvectors.data = self._compute_autorcorrelation().symeig(True)
        self.eigenvalues.data, idx = self.eigenvalues.sort(descending=True)
        # correct numerical error, matrix must be positivly semi-definitie
        self.eigenvalues[self.eigenvalues < 0] = 0
        self.eigenvectors.data = self.eigenvectors[:, idx]

    def _reset_autorcorrelation(self):
        self.autorcorrelation_matrix.data = torch.zeros(self.autorcorrelation_matrix.shape).to(self.autorcorrelation_matrix.device)
        self.seen_samples.data = torch.zeros(self.seen_samples.shape).to(self.autorcorrelation_matrix.device)

    def _compute_pca_matrix(self):
        if self.verbose:
            print('computing autorcorrelation for Linear')
            #print('Mean pre-activation vector:', self.mean)
        percentages = self.eigenvalues.cumsum(0) / self.eigenvalues.sum()
        eigen_space = self.eigenvectors[:, percentages < self.threshold]
        if eigen_space.shape[1] == 0:
            eigen_space = self.eigenvectors[:, :1]
            print(f'Detected singularity defaulting to single dimension {eigen_space.shape}')
        elif self.threshold - (percentages[percentages < self.threshold][-1]) > 0.02:
            print(f'Highest cumvar99 is {percentages[percentages < self.threshold][-1]}, extending eigenspace by one dimension for eigenspace of {eigen_space.shape}')
            eigen_space = self.eigenvectors[:, :eigen_space.shape[1]+1]

        if self.verbose:
            print(f'Saturation: {round(eigen_space.shape[1] / self.eigenvalues.shape[0], 4)}%', 'Eigenspace has shape', eigen_space.shape)
        self.transformation_matrix: torch.Tensor = eigen_space.matmul(eigen_space.t())
        self.reduced_transformation_matrix: torch.Tensor = eigen_space
        self.reversed_transformaton_matrix: torch.Tensor = self.eigenvectors[:, eigen_space.shape[1]-1:] @ self.eigenvectors[:, eigen_space.shape[1]-1:].t()

    def forward(self, x):
        trans_mat = None
        if self.gradient_epoch < self.epoch:
            if self.boosted:
                trans_mat = self.reversed_transformaton_matrix
            else:
                trans_mat = self.transformation_matrix
        if self.training:
            self.pca_computed = False
            self._update_autorcorrelation(x)
        else:
            if not self.pca_computed:
                self.epoch += 1
                if 50 < self.epoch:
                    self.boosted = not self.boosted
                self._compute_autorcorrelation()
                self._compute_eigenspace()
                self._compute_pca_matrix()
                self.pca_computed = True
                self._reset_autorcorrelation()
            if self.boosted:
                trans_mat = self.reversed_transformaton_matrix
            if self.keepdim:
                trans_mat = self.reduced_transformation_matrix
        return LinearPCALayerFunction.apply(x, trans_mat)


class Conv2DPCALayer(LinearPCALayer):

    def __init__(self, in_filters, threshold: float = 0.99, verbose: bool = True, gradient_epoch_start: int = 1, boosted: bool = False
                 ):
        super(Conv2DPCALayer, self).__init__(in_features=in_filters, threshold=threshold, keepdim=True, verbose=verbose, gradient_epoch_start=gradient_epoch_start, boosted=boosted)
        if verbose:
            print('Added Conv2D PCA Layer')
        self.pca_conv = torch.nn.Conv2d(in_channels=in_filters,
                                        out_channels=in_filters,
                                        kernel_size=1, stride=1, bias=False)

    def _compute_pca_matrix(self):
        if self.verbose:
            print('computing autorcorrelation for Conv2D')
        super()._compute_pca_matrix()
        # unsequeeze the matrix into 1x1xDxD in order to make it behave like a 1x1 convolution
        if self.boosted:
            weight = torch.nn.Parameter(self.reversed_transformaton_matrix.unsqueeze(2).unsqueeze(3))
        else:
            weight = torch.nn.Parameter(self.transformation_matrix.unsqueeze(2).unsqueeze(3))
        self.pca_conv.weight = weight

    def forward(self, x):
        conv = None
        if self.training:
            self.pca_computed = False
            swapped: torch.Tensor = x.permute([1, 0, 2, 3])
            flattened: torch.Tensor = swapped.flatten(1)
            reshaped_batch: torch.Tensor = flattened.permute([1, 0])
            self._update_autorcorrelation(reshaped_batch)
        else:
            if not self.pca_computed:
                self.epoch += 1
                if 50 < self.epoch:
                    self.boosted = not self.boosted
                self._compute_autorcorrelation()
                self._compute_eigenspace()
                self._compute_pca_matrix()
                self._reset_autorcorrelation()
                #ctx.save_for_backward(self.backwards_convolution)
                self.pca_computed = True
                conv = self.pca_conv
        if self.gradient_epoch is not None and self.gradient_epoch < self.epoch:
            conv = self.pca_conv
        return Conv2DPCALayerFunction.apply(x, conv)
