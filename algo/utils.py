import ray
import numpy as np
from scipy.stats import multivariate_normal as mvn


@ray.remote
def create_shared_gaussian_noise(seed=12345):
    np.random.seed(seed)
    count = 5000000
    noise = np.random.RandomState(seed).randn(count).astype(np.float32)
    return noise


class SharedNoiseTable(object):
    def __init__(self, noise, seed=123):
        np.random.seed(seed)
        self.rg = np.random.RandomState(seed)
        self.noise = noise
        #  assert self.noise.dtype == np.float64

    def get(self, i, dim):
        return self.noise[i:i + dim]

    def sample_index(self, dim):
        return self.rg.randint(0, len(self.noise) - dim + 1)

    def get_delta(self, dim):
        idx = self.sample_index(dim)
        return idx, self.get(idx, dim)
    
# OpenAI-ES
def compute_ranks(x):
    """Returns ranks in [0, len(x))

    Note: This is different from scipy.stats.rankdata, which returns ranks in
    [1, len(x)].
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks

def compute_centered_ranks(x):
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= 0.5
    return y

def sqrt_matrix(A, cov_mode='full'):
    """Compute square root of a matrix."""
    if cov_mode == 'full':
        eigvals, eigvecs = np.linalg.eigh(A)
        sqrt_A = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    else:
        sqrt_A = np.sqrt(A)
    return sqrt_A

def check_matrix_pos(A_any):
    """Check if a matrix is positive definite."""
    eigvals, eigvecs = np.linalg.eigh(A_any)
    eigvals_pos = eigvals.clip(min=1e-4, max=1)
    A_pos = eigvecs @ np.diag(eigvals_pos) @ eigvecs.T
    return A_pos

def calculate_isratio(old_mean, old_cov, curr_mean, curr_cov, sample):
    """Calculate importance sampling ratio."""
    old_prob = mvn.pdf(sample, mean=old_mean, cov=old_cov)
    curr_prob = mvn.pdf(sample, mean=curr_mean, cov=curr_cov)
    return curr_prob / old_prob

def calculate_meancovw(curr_mean, curr_cov, curr_cov_inv, sample, cov_type):
    """Calculate mean and cov w."""
    mean_w = np.linalg.solve(curr_cov, sample - curr_mean)
    if cov_type == 'full':
        cov_W = -0.5 * curr_cov_inv + 0.5 * np.outer(mean_w, mean_w)
    elif cov_type == 'diag':
        curr_cov_diag = np.diagonal(curr_cov)
        cov_W_diag = -0.5 / curr_cov_diag + 0.5 * mean_w * mean_w
        cov_W = np.diag(cov_W_diag)
    else:
        cov_W = np.zeros(curr_cov.shape)
    return mean_w, cov_W