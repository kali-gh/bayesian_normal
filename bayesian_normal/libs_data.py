import logging
import numpy as np
import random

random.seed(42)
np.random.seed(42)

logger = logging.getLogger(__name__)

def build_dummy_dataset(inputs):
    """
    Builds a dummy dataset similar to Bishop 2006, section 3.1

    :param inputs: input parameters , see test_known_var.py
    :return: dictionary
        prior_mean : prior means
        prior_cov : prior covariance
        X : X observations
        y : y observations as a linear function of X
    """

    logger.info("INPUTS")
    logger.info(
        f"{inputs['w0_true']=}\n"
        f"{inputs['w1_true']=}\n"
        f"{inputs['prior_alpha']=}\n"
        f"{inputs['precision_y']=}\n"
    )

    logger.info("Setting parameters")

    std_dev_y = np.sqrt(1 / inputs['precision_y'])
    X = np.random.uniform(-1, 1, 20)

    data = {
        'prior_mean': np.reshape(np.array([0, 0]), (-1, 1)),
        'prior_cov': (1 / inputs['prior_alpha']) * np.identity(2),

        'X': X,
        'y': inputs['w0_true'] + inputs['w1_true'] * X + np.random.normal(0, scale=std_dev_y)
    }
    logger.info("Data")
    logger.info(data)

    return data