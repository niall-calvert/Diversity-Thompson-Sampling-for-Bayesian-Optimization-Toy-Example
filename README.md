## Diversity-Filtered Batch Thompson Sampling

This project implements a simple modification of batch Thompson sampling for Bayesian optimization.

The baseline method follows the BoTorch Thompson sampling tutorial. At each optimization step, a Gaussian process surrogate model is fit to the observed data. A large Sobol candidate set is generated over the normalized input domain, and `MaxPosteriorSampling` is used to select a batch of promising candidate points.

The modification introduced here is to oversample Thompson candidates before choosing the final evaluation batch. Instead of directly requesting `batch_size` points from Thompson sampling, the method first requests a larger pool of candidates:

```python
X_pool = thompson_sampling(X_cand, num_samples=pool_size)
