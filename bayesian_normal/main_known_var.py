import logging
import numpy as np
import math
import matplotlib.pyplot as plt
import random

random.seed(42)
np.random.seed(42)

from scipy.stats import multivariate_normal

from bayes_normal_known_var import BayesNormal
from libs_dist import Normal

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

#### parameters
prior_alpha = 2.0
prior_mean = np.reshape(np.array([0, 0]), (-1, 1))
prior_cov = (1/prior_alpha)*np.identity(2)

w0_true = -0.7
w1_true = 0.5
precision_y = 25
std_dev_y = np.sqrt(1/precision_y)
X = np.random.uniform(-1,1,20)
y = w0_true + w1_true*X + np.random.normal(0, scale=std_dev_y)

logger.info("PARAMETERS")
logger.info(
    f"{prior_alpha=}\n"
    f"{prior_mean=}\n"
    f"{prior_cov=}\n"

    f"{w0_true=}\n"
    f"{w1_true=}\n"

    f"{precision_y=}\n"
    f"{std_dev_y=}\n"
    f"{X=}\n"
    f"{y=}"
)


##### main script
logger.info("Initializing prior")
prior = Normal(mean=prior_mean, cov=prior_cov)
logger.info(prior)

def model(W_array, X_array):

    return W_array[0] + W_array[1]*X_array

bn = BayesNormal(variance=1, prior_mean=prior_mean, prior_cov=prior_cov, intercept=True)


bn.add_observations(X, y)

logger.info('posterior')
logger.info(bn.posterior)
