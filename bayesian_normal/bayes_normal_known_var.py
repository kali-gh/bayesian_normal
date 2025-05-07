import numpy as np
import math
import logging

logger = logging.getLogger(__name__)

from typing import List
from scipy.stats import multivariate_normal

from .libs_dist import Normal
from sklearn.base import BaseEstimator


class BayesNormal:
    """
    A class for multivariate sequential Bayesian estimation
    We follow the development and notation of Bishop, 2006 - see section 3.1
    """

    def __init__(
            self,
            variance,
            prior_mean : np.array,
            prior_cov : np.array,
            intercept=False):
        """
        Initialize the BayesNormal class which assumes the variance is known and can learn sequentially

        Args:
            variance (float):
                variance of the resposne, representing the noise. internally we use beta=1/variance
                assumed known
            prior_mean (np.array):
                prior mean as an array [Px1]
            prior_cov (np.array)
                prior covariance as an array [PxP]
            intercept
                include intercept in model (phi matrix is left padded with 1's)
        """

        self.beta = 1/variance

        self.m_0 = prior_mean # notation bishop 2006, p153
        self.S_0 = prior_cov # notation bishop 2006, p153

        self.intercept = intercept

        self.prior = Normal(mean=prior_mean, cov=prior_cov)
        self.posterior = None

    def add_observations(
            self,
            X : np.array | List,
            y : np.array | List):
        """
        Add observations to the model and update the posterior

        Args:
            X (np.array | List):
                input data, either a single observation or a list of observations
            y (np.array | List):
                output data, either a single observation or a list of observations
        """

        if type(X) == list:
            X = np.array(X)
        else:
            pass

        if type(y) == list:
            y = np.array(y)
        else:
            pass

        if len(X.shape) == 1:  # 1d array
            X = X.reshape((-1, 1))  # Nx1
        else:
            pass

        if len(y.shape) == 1:  # 1d array
            y = y.reshape((-1, 1))  # Nx1
        else:
            pass

        if self.intercept:
            ones = np.reshape(np.ones(X.shape[0]),  (-1, 1))
            phi = np.hstack([ones, X])
        else:
            phi = X

        logger.info(f'phi shape : {phi.shape}')

        if self.posterior is None:
            prior_mean = self.m_0
            prior_cov = self.S_0
        else:
            prior_mean = self.posterior.mean
            prior_cov = self.posterior.cov


        S_0_inv = np.linalg.inv(prior_cov)
        S_N_inv = S_0_inv + self.beta * phi.T @ phi
        S_N = np.linalg.inv(S_N_inv)

        m_N = S_N @ (
            (S_0_inv @ prior_mean) +
            (self.beta * phi.T @ y)
        )

        self.posterior = Normal(mean=m_N, cov=S_N)


class BayesNormalEstimator(BaseEstimator):
    def __init__(
            self,
            *,
            variance,
            prior_mean,
            prior_cov):

        self.variance = variance
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov

        self.bn = BayesNormal(
            variance=variance,
            prior_mean=prior_mean,
            prior_cov=prior_cov)

    def fit(self, X, y=None):

        self.is_fitted_ = True
        self.bn.add_observations(X, y)

        return self

    def predict(self, X):
        return X @ self.bn.posterior.mean