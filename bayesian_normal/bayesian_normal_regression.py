


import logging
import numpy as np
from scipy.stats import invgamma
from scipy.stats import multivariate_normal
from dataclasses import dataclass

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@dataclass
class NormalInverseGammaDistribution():
    mu : np.array
    sigma : float
    lam: np.array
    a: int
    b : int
    cov: np.array = None

class BayesianNormalRegression():
    
    def __init__(self):
        self.posterior = None
    
    def fit(self, X, y, prior: NormalInverseGammaDistribution):
        
        logger.debug(prior)
        
        N = y.shape[0]
        
        self.prior = prior
        
        # posterior lam_n
        lam_n = np.matmul(X.T, X) + prior.lam
        
        # posterior mu_n
        temp = np.matmul(prior.lam, prior.mu) + np.matmul(X.T, y)
        logger.debug(f"temp: {temp.shape}")
        cov = np.linalg.inv(lam_n)
        mu_n = np.matmul(cov, temp)
        
        # posterior for inverse gamma
        a_n = prior.a + 0.5 * N
        b_n = prior.b + 0.5 * (np.dot(y.T, y) + np.dot(prior.mu.T, prior.lam.T @ prior.mu) - np.dot(mu_n.T, lam_n.T @ mu_n) )
        
        # Consolidate posterior distribution
        self.posterior = NormalInverseGammaDistribution(mu=mu_n, sigma=prior.sigma, lam=lam_n, a=a_n, b=b_n, cov=cov)
        
        return self.posterior
                

    def sample_posterior(self):

        n_dim = self.posterior.mu.shape[0]
        
        # Sample sigma2, and beta conditional on sigma2
        sigma2_s =  self.posterior.b * invgamma.rvs(self.posterior.a) 
        
        logger.debug(sigma2_s)
        
        
        cov =  sigma2_s * self.posterior.cov
        
        logger.debug(f"shape posterior {cov.shape}")

        try:
            beta_s = [
                np.random.multivariate_normal(self.posterior.mu.reshape((-1,)), cov) 
            ]
        except np.linalg.LinAlgError as e:
            logger.error(e)

        return beta_s, sigma2_s