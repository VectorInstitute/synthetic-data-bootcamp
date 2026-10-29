"""Empirical Gaussian copula with tied ranks and regularized dependence."""
from dataclasses import dataclass
from statistics import NormalDist
import numpy as np

NORMAL = NormalDist()


def positive_correlation(matrix):
    """Project a symmetric estimate to a positive definite correlation matrix."""
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2)
    repaired = (vectors * np.maximum(values, 1e-8)) @ vectors.T
    scale = np.sqrt(np.diag(repaired))
    return repaired / np.outer(scale, scale)


@dataclass
class GaussianCopula:
    marginals: list
    correlation: np.ndarray
    imputed_count: int
    n: int

    @classmethod
    def fit(cls, matrix, shrinkage=0.15, prior=None):
        x = np.asarray(matrix, dtype=float).copy()
        if x.ndim != 2 or len(x) < 2:
            raise ValueError('At least two rows and a 2D matrix are required')
        if not 0 <= shrinkage <= 1:
            raise ValueError('shrinkage must be between zero and one')
        missing = ~np.isfinite(x)
        marginals, scores = [], []
        for j in range(x.shape[1]):
            observed = np.sort(x[~missing[:, j], j])
            if not len(observed):
                raise ValueError(f'Column {j} has no observed values')
            marginals.append(observed)
            x[missing[:, j], j] = np.median(observed)
            _, inverse, counts = np.unique(x[:, j], return_inverse=True, return_counts=True)
            ranks = np.cumsum(counts) - (counts - 1) / 2
            u = (ranks[inverse] - 0.5) / len(x)
            scores.append(np.array([NORMAL.inv_cdf(float(p)) for p in u]))
        z = np.column_stack(scores)
        sd = z.std(axis=0)
        standardized = (z - z.mean(axis=0)) / np.where(sd > 1e-12, sd, 1)
        corr = standardized.T @ standardized / len(x)
        np.fill_diagonal(corr, 1)
        prior = np.eye(x.shape[1]) if prior is None else np.asarray(prior)
        corr = positive_correlation((1 - shrinkage) * corr + shrinkage * prior)
        return cls(marginals, corr, int(missing.sum()), len(x))

    def sample(self, n, seed):
        if n < 1:
            raise ValueError('n must be positive')
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((n, len(self.marginals))) @ np.linalg.cholesky(self.correlation).T
        u = np.array([NORMAL.cdf(float(v)) for v in z.ravel()]).reshape(z.shape)
        return np.column_stack([np.quantile(m, u[:, j]) for j, m in enumerate(self.marginals)])

    def to_dict(self):
        return dict(n=self.n, imputed_count=self.imputed_count,
                    marginals=[m.tolist() for m in self.marginals],
                    correlation=self.correlation.tolist())
