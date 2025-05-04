import logging

from bayes_normal_known_var import BayesNormal
from libs_dist import Normal
from libs_data import build_dummy_dataset

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

inputs = {
    'prior_alpha' : 2.0,
    'w0_true' : -0.7,
    'w1_true' : 0.5,
    'precision_y' : 25,
    'N' : 100
}

logger.info("Setting inputs")
logger.info(inputs)

data = build_dummy_dataset(
    inputs
)

##### main script
logger.info("Initializing prior")
prior = Normal(mean=data['prior_mean'], cov=data['prior_cov'])
logger.info(prior)

bn = BayesNormal(variance=1/inputs['precision_y'], prior_mean=data['prior_mean'], prior_cov=data['prior_cov'], intercept=True)

bn.add_observations(data['X'], data['y'])

logger.info('posterior')
logger.info(bn.posterior)
