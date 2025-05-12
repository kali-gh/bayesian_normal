import logging
import numpy as np

import libs_data

from bayesian_normal.bayes_normal_known_var import BayesNormal
from bayesian_normal.bayes_normal_known_var import BayesNormalEstimator
from bayesian_normal.libs_dist import Normal
from libs_data import get_bishop_data

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

"""
Limited set of tests on sequential learning
    - Checks if we can match Bishop on the dummy problem
    - Checks if posterior estimates are close to true means in a dummy 2d no intercept problem
    - Checks if BayesNormal matches the estimator for the 2d no intercept problem
    - Checks if learning sequentially in batches yields the same posterior as if we learn in 1-shot
"""

#bishop
inputs_bishop = {
    'prior_alpha': 2.0,
    'w0_true': -0.7,
    'w1_true': 0.5,
    'precision_y': 25,
    'N': 100
}
data_bishop = get_bishop_data(
    inputs_bishop
)

# 2d dummy
w_true_2d = np.array([-0.7, 0.5]).reshape(-1, 1) # true 2d, inited only once for testing
inputs_2d, data_2d = libs_data.get_2d_no_intercept(w_true=w_true_2d, N=100)

def test_bishop_data_intercept():
    """
    Checks if the library can estimate posterior means the Bishop problem using the BayesNormal object
    within a generous tolerance w/100 datapoints

    :after: throws if assertion on the mean fails
    """

    logger.info("Initializing prior")
    prior = Normal(mean=data_bishop['prior_mean'], cov=data_bishop['prior_cov'])
    logger.info(prior)

    bn = BayesNormal(
        variance=1 / inputs_bishop['precision_y'],
        prior_mean=data_bishop['prior_mean'],
        prior_cov=data_bishop['prior_cov'],
        intercept=True)

    bn.add_observations(data_bishop['X'], data_bishop['y'])

    logger.info('posterior')
    logger.info(bn.posterior)

    ZERO_TOL = 0.001

    assert np.abs(bn.posterior.mean[0] - -0.68210729) < ZERO_TOL
    assert np.abs(bn.posterior.mean[1] - 0.49895389) < ZERO_TOL

    #Normal(
    #    mean=
    #    [[-0.68210729]
    #     [0.49895389]],
    #    cov=
    #    [[4.03724031e-04 6.78590128e-05]
    #     [6.78590128e-05 1.13874904e-03]]
    #)

def get_2d_no_intercept():
    bne = BayesNormalEstimator(
        variance=1 / inputs_2d['precision_y'],
        prior_mean=data_2d['prior_mean'],
        prior_cov=data_2d['prior_cov'])

    bne = bne.fit(data_2d['X'], data_2d['y'])

    #Normal(
    #    mean=array([[-0.70985217],
    #                   [0.50629017]]),
    #    cov=array([[1.12024084e-03, -8.29780573e-05],
    #          [-8.29780573e-05, 1.24170529e-03]]))

    return bne


def test_2d_no_intercept():
    """
    Checks if the library can estimate posterior means for a no intercept 2d problem using the BayesNormalEstimator
    object
        within a wide margin for the mean (0.05, since 100 data points only)
        within a reasonably tight tolerance (1e-3)
    :return: fitted estimator (to use as reference in other tests)
    """

    bne = get_2d_no_intercept() # see this function for reference estimates at set seed

    OK_TOLERANCE_MEAN = 0.05 # mean estimates with 100 data points are still pretty noisy

    assert np.abs(bne.bn.posterior.mean[0] - w_true_2d[0]) < OK_TOLERANCE_MEAN, 'mean[0] does not match'
    assert np.abs(bne.bn.posterior.mean[1] - w_true_2d[1]) < OK_TOLERANCE_MEAN, 'mean[1] does not match'

    OK_TOLERANCE_COV = 1E-3
    assert np.abs(bne.bn.posterior.cov[0,0] - 1.12879383e-03) < OK_TOLERANCE_COV,  'cov[0,0] does not match'
    assert np.abs(bne.bn.posterior.cov[0, 1] - -2.87881813e-05) < OK_TOLERANCE_COV,  'cov[0,0] does not match'
    assert np.abs(bne.bn.posterior.cov[1, 0] - -2.87881813e-05) < OK_TOLERANCE_COV,  'cov[0,0] does not match'
    assert np.abs(bne.bn.posterior.cov[1, 1] - 1.28031165e-03) < OK_TOLERANCE_COV,  'cov[0,0] does not match'


def test_2d_bayes_normal():
    """
    Checks if BayesNormal gives the same posterior estimates as the BayesNormalEstimator
    :return: fitted estimator (to use as reference in other tests)
    """

    bn = BayesNormal(
        variance=1 / inputs_2d['precision_y'],
        prior_mean=data_2d['prior_mean'],
        prior_cov=data_2d['prior_cov'])

    bn.add_observations(data_2d['X'], data_2d['y'])

    reference_posterior = get_2d_no_intercept()

    ZERO_TOL = 1e-5
    for k in range(0, 2):
        assert np.abs(bn.posterior.mean[k] - reference_posterior.bn.posterior.mean[k]) < ZERO_TOL

    for i in range(0, 2):
        for j in range(0, 2):
            assert np.abs(bn.posterior.cov[i, j] - reference_posterior.bn.posterior.cov[i, j]) < ZERO_TOL


def test_2d_no_intercept_sequential():
    """
    Checks if sequential estimation in batches of 20 points gets the same result as 1-shot estimation

    Uses BNE 1-shot estimator as reference

    :after: throws if assertion on the mean or covariance fails
    """

    bn = BayesNormal(
        variance=1 / inputs_2d['precision_y'],
        prior_mean=data_2d['prior_mean'],
        prior_cov=data_2d['prior_cov'])

    MIN = 0
    MAX = 100
    STEP = 20
    for k in range(MIN, MAX, STEP):
        bn.add_observations(data_2d['X'][k:k + STEP, :], data_2d['y'][k:k + STEP, :])

    reference_posterior = get_2d_no_intercept()

    ZERO_TOL = 1e-5
    for k in range(0, 2):
        assert np.abs(bn.posterior.mean[k] - reference_posterior.bn.posterior.mean[k]) < ZERO_TOL

    for i in range(0, 2):
        for j in range(0, 2):
            assert np.abs(bn.posterior.cov[i, j] - reference_posterior.bn.posterior.cov[i, j]) < ZERO_TOL

    # Needs to match exactly when we learn sequentially instead of in 1-shot, which it does
    # Normal(mean=array([[-0.70985217],
    #                   [0.50629017]]),
    # cov =array([[1.12024084e-03, -8.29780573e-05],
    #  [-8.29780573e-05, 1.24170529e-03]]))
