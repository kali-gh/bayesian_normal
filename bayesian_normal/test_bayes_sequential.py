import logging
import numpy as np
import math
import matplotlib.pyplot as plt
import random

from scipy.stats import multivariate_normal

from libs_dist import Normal

np.random.seed(42)
random.seed(42)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#### parameters
prior_alpha = 2.0
prior_mean = np.array([0, 0])
prior_cov = (1 / prior_alpha) * np.identity(2)

beta=2

w0_true = -0.7
w1_true = 0.5
precision_y = 25
std_dev_y = np.sqrt(1 / precision_y)
noise_y = np.random.normal(0, scale=std_dev_y)
x = np.random.uniform(-1, 1, 20)
y = w0_true + w1_true * x + noise_y

logger.info("PARAMETERS")
logger.info(
    f"{prior_alpha=}\n"
    f"{prior_mean=}\n"
    f"{prior_cov=}\n"

    f"{w0_true=}\n"
    f"{w1_true=}\n"

    f"{precision_y=}\n"
    f"{std_dev_y=}\n"
    f"{noise_y=}\n"
    f"{x=}\n"
    f"{y=}"
)

##### main script
logger.info("Initializing prior")
prior = Normal(mean=prior_mean, cov=prior_cov)
logger.info(prior)


def model(w_params, x_values):
    return w_params[0] + w_params[1]*x_values

def calc_likelihood(y_values, x_values, w_params, callable_mean):

    N = len(x_values)

    e_d_w = 0.0
    for yv,xv in zip(y_values,x_values):
        e_d_w += (yv-(callable_mean(w_params=w_params, x_values=xv)))**2

    e_d_w = e_d_w*0.5

    likelihood = 0.5*N*np.log(beta) - 0.5*N*np.log(2*math.pi) - beta * e_d_w
    return likelihood

w0 = np.linspace(-1, 1, 20)
w1 = np.linspace(-1, 1, 20)


def calculate(xv, yv, prior, in_posterior, use_posterior_as_prior):

    ll = np.zeros(shape=(20, 20))

    posterior = np.zeros(shape=(20, 20))

    for i, w0c in enumerate(w0):
        for j, w1c in enumerate(w1):
            w_params = (w0c, w1c)

            ll[i, j] = calc_likelihood(y_values=yv, x_values=xv, w_params=w_params,  callable_mean=model)

            if not use_posterior_as_prior:
                prior[i, j] = multivariate_normal.pdf([w0c,w1c], mean=prior_mean, cov=prior_cov)
            else:
                prior[i, j] = in_posterior[i,j]

            posterior[i, j] = np.exp(ll[i, j]) * prior[i, j]

    posterior = posterior / np.sum(posterior)

    return ll, prior, posterior


from pathlib import Path

Path("plots").mkdir(exist_ok=True)
Path("temp").mkdir(exist_ok=True)

prior = np.zeros(shape=(20, 20))
posterior = np.zeros(shape=(20, 20))

idx = 0
for xv, yv in zip(x, y):

    logger.info(f"processing {idx=}: {xv=} {yv=}")
    use_posterior_as_prior = (idx > 0)
    ll, prior, posterior = calculate(
        xv=[xv],
        yv=[yv],
        prior=prior,
        in_posterior=posterior,
        use_posterior_as_prior=use_posterior_as_prior
    )

    plt.figure()
    plt.contourf(w0, w1, prior)
    plt.colorbar()
    plt.savefig(f'temp/{idx=}_prior.png')

    plt.figure()
    plt.contourf(w0, w1, posterior)
    plt.colorbar()
    plt.savefig(f'temp/{idx=}_posterior.png')

    idx += 1

    if idx == 20:
        break