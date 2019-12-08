from pca_layers import LinearPCALayer
from delve.torch_utils import TorchCovarianceMatrix
import numpy as np
import torch

np.random.seed(1)
N = 100000
b1 = np.random.rand(N)
b2 = np.random.rand(N)
X = np.column_stack([b1, b2])
X -= X.mean(axis=0)
fact = N - 1
by_hand = np.dot(X.T, X.conj()) / fact

using_cov = np.cov(b1, b2)
pca_layer = LinearPCALayer(2, threshold=.99, centering=True)
pca_layer(torch.from_numpy(X))
pca_layer.eval()
cov_mtx = pca_layer._compute_autorcorrelation().cpu().numpy()
tcm = TorchCovarianceMatrix()
tcm.update(torch.from_numpy(X).to('cuda:0'))
cmt = tcm.fix(True).cpu().numpy()

print('Manual')
print(by_hand)
print('Numpy')
print(cov_mtx)
print('Torch-PCA-Layer')
print(cov_mtx)
print('Delve Torch Covariance Approximator')
print(cmt)
assert np.allclose(cov_mtx, using_cov, rtol=0.0001)
assert np.allclose(cmt, using_cov, rtol=0.0001)