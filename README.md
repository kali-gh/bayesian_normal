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

```
python test_known_var.py
```
