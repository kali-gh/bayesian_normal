import logging
import numpy as np
import random

logger = logging.getLogger(__name__)

random.seed(42)
np.random.seed(42)

def get_bishop_data(
        inputs):
    """
    Builds a dummy dataset similar to Bishop 2006, section 3.1

    :param inputs: input parameters , see test_known_var.py
        w0_true : true intercept
        w1_true : true intercept
        prior_alpha : prior alpha scaling factor - see Bishop
        precision_y : inverse variance of generating process for y - see Bishop
        N : number of datapoints to generate
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
    X = np.random.uniform(-1, 1, inputs['N'])

    data = {
        'prior_mean': np.reshape(np.array([0, 0]), (-1, 1)),
        'prior_cov': (1 / inputs['prior_alpha']) * np.identity(2),

        'X': X,
        'y': inputs['w0_true'] + inputs['w1_true'] * X + np.random.normal(0, scale=std_dev_y)
    }
    logger.info("Data")
    logger.info(data)

    return data


def get_dN_data(
        inputs):
    """
    Builds a dummy dataset similar to Bishop 2006, section 3.1

    :param inputs: input parameters , see test_known_var.py
        w_true - shape (M, 1) : true intercept,
        prior_alpha - float : prior alpha scaling factor - see Bishop
        precision_y - float: inverse variance of generating process for y - see Bishop
    :return: dictionary
        prior_mean : prior means
        prior_cov : prior covariance
        X : X observations, size N (number of observations) by M ( number of parameters)
        y : y observations as a linear function of X
    """

    logger.info("Inputs for dummy data generation")
    logger.info(
        f"{inputs['w_true']=}\n"
        f"{inputs['prior_alpha']=}\n"
        f"{inputs['precision_y']=}\n"
    )

    logger.info("Setting parameters")
    M = inputs['w_true'].shape[0]
    shape_X = (inputs['N'], M)

    logger.info(f"Number of paramters : {M=}")
    logger.info(f"Building dummy dataset shape : {shape_X=}")

    X = np.random.uniform(-1, 1, shape_X)
    logger.info(f"X shape : {X.shape}")

    prior_mean = np.zeros((M, 1))

    std_dev_y = np.sqrt(1 / inputs['precision_y'])
    data = {
        'prior_mean': prior_mean,

        'prior_cov': (1 / inputs['prior_alpha']) * np.identity(M),

        'X': X,

        'y': X @ inputs['w_true'] + np.random.normal(0, scale=std_dev_y)
    }

    logger.info("Data")

    logger.info(f"Shape of prior mean: {prior_mean.shape}")
    logger.info(f"Shape of prior covariance: {data['prior_cov'].shape}")
    logger.info(f"Shape of X: {X.shape}")
    logger.info(f"Shape of y: {data['y'].shape}")

    return data

def get_2d_no_intercept(w_true, N):

    inputs = {
        'prior_alpha': 2.0,
        'w_true': w_true,
        'precision_y': 25,
        'N': N
    }

    data = get_dN_data(
        inputs=inputs
    )
    logger.info(f"data keys : {data.keys()}")

    return inputs, data
