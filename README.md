### Background

There are very few python libraries which implement Bayesian methods for sequential learning in a multivariate setting. 

The goal of this library is to take steps towards doing that. Our end goal for the library is that it have two parts:
1) Able to learn from sequential data where the generation process has an unknown mean but a known variance
2) Same as 1), for case of unknown mean and unknown variance

### Repo overview

In the current state of the library, we implement two main scripts:

- bayes_sequential : sequential learning for a 1-d regression problem as shown in Bishop 2006, page 155.
- main_known_var : implement general solution as shown in Bishop 2006 for case of unknown mean and known variance

### Set up 

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Testing

As an example problem we estimate parameters for a simple linear regression as in Bishop, 2006, section 3.1. 

In that setting the outcome y is a linear function of X with an intercept as:

$$
y_n = w_0 + w_1 x_n 
$$

and our goal is to discover the parameters for this function. Assuming that the true data generating process is 

$$
w_0 = -0.7 
$$

$$
w_1 = 0.5
$$

We can test the library by running the following commands:

```python
np.random.seed(42)

import numpy as np

inputs = {
    'prior_alpha' : 2.0,
    'w0_true' : -0.7,
    'w1_true' : 0.5,
    'precision_y' : 25,
    'N' : 100
}

std_dev_y = np.sqrt(1 / inputs['precision_y'])
X = np.random.uniform(-1, 1, inputs['N'])

data = {
    'X': X,
    'y': inputs['w0_true'] + inputs['w1_true'] * X + np.random.normal(0, scale=std_dev_y)
}
```

Note in the above we have set the variance of $y$ to be $1/\lambda_y=1/25$ where $\lambda_y$ is the precision or inverse variance. 
This controls the random noise in y in the true generating process. We have also set a numpy random seed to make the outputs reproducible.  

Next to estimate the model we need to set prior parameters for it as 

```python
data.update({
    'prior_alpha' : 2
})

data.update({
    'prior_mean': np.reshape(np.array([0, 0]), (-1, 1)),
    'prior_cov': (1 / data['prior_alpha']) * np.identity(2)
})
```

This sets the prior mean to zero for each parameter and the prior covariance to the identity matrix divided by two (the parameter $\alpha$)

We can next set the prior for estimation using

```python
from libs_dist import Normal
prior = Normal(mean=data['prior_mean'], cov=data['prior_cov'])
```

The prior is then defined as follows:
```python
Normal(
    mean= array(
           [[0],[0]]),
       
    cov=array(
           [[0.5, 0. ],
           [0. , 0.5]]
       ))
```

We can form the estimator in the following way:

```python
from bayes_normal_known_var import BayesNormal
bn = BayesNormal(
    variance=1/inputs['precision_y'], 
    prior_mean=data['prior_mean'], 
    prior_cov=data['prior_cov'], 
    intercept=True)
```

Next we can inform the estimator of new observations using

```python
bn.add_observations(data['X'], data['y'])
```

and examine the posterior:
```python
Normal(
    mean=array(
        [[-0.68210729],
         [ 0.49895389]]), 
    cov=array(
        [[4.03724031e-04, 6.78590128e-05],
        [6.78590128e-05, 1.13874904e-03]]
    ))
```

so that after 20 observations the intercept is estimated as $-0.682$ vs the ground truth of $-0.700$ and 
the slope is $0.499$ vs. the ground truth of $0.500$


This is also encapsulated in the following script:

```
python test_known_var.py
```


### Scikit-learn Estimator

The package also implements a scikit-learn estimator for fit and predict

Forming the estimator :
```python
from bayesian_normal.bayes_normal_known_var import BayesNormalEstimator

clf = BayesNormalEstimator(
    variance=1/inputs['precision_y'],
    prior_mean=data['prior_mean'],
    prior_cov=data['prior_cov'],
    intercept=True
)
```

We can fit the estimator with

```python
clf.fit(data['X'], data['y'])
```

and can examine the posterior with
```python
clf.bn.posterior

Normal(
    mean=array(
      [[-0.68210729],
       [ 0.49895389]]
    ), 
    cov=array(
        [[4.03724031e-04, 6.78590128e-05],
        [6.78590128e-05, 1.13874904e-03]]
    ))
```

### Prediction

We can build predictions with the estimator by calling
```python
clf.predict(data['X'])

    0,-0.8073046805998443
    1,-0.23233597527166444
    2,-0.45059872909983767
    ...
```

These are the posterior mean estimates after the fit. 

We can also Thompson sample from the posterior distribution predictions by calling

```python
clf = BayesNormalEstimator(
    variance=1/inputs['precision_y'],
    prior_mean=data['prior_mean'],
    prior_cov=data['prior_cov'],
    intercept=True,
    predict_thompson_sample=True
)
clf.fit(data['X'], data['y'])
clf.predict(data['X'])
    -0.803835208966934
    -0.2406691453895312
    -0.45445152129956784
    ...
```

In this case the predictions sample parameters from the posterior and use these in forming their prediction.



### Building

To build as wheel
```
python setup.py bdist_wheel
```

### Testing

After installing to venv

Limited set of tests on sequential learning
1. Checks if we can match Bishop on the dummy problem
2. Checks if posterior estimates are close to true means in a dummy 2d no intercept problem
3. Checks if BayesNormal matches the estimator for the 2d no intercept problem
4. Checks if learning sequentially in batches yields the same posterior as if we learn in 1-shot

```python
pytest tests/test_known_var.py
```

To check the Bishop problem separately

```
python tests/test_bayes_sequential.py 
```


