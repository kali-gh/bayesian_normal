import numpy as np

from dataclasses import dataclass


@dataclass
class Normal:
    mean : np.array  # mean for the multivariate normal distribution, of length N
    cov : np.array # covariance of multivariate normal, of shape NxN

    def __str__(self):
        return f"\nNormal(\nmean=\n{self.mean},\n cov=\n{self.cov}\n)"

