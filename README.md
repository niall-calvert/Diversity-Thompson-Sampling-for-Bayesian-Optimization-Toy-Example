## Diversity Filtered Batch Thompson Sampling

This project builds on the BoTorch batch Thompson sampling tutorial for Bayesian optimization. In the baseline method, a Gaussian process model is fit to the observed data, a Sobol candidate set is generated, and `MaxPosteriorSampling` selects the next batch of points to evaluate using cholesky decomposition.

My modification oversamples Thompson candidates before choosing the final batch. Instead of sampling only `batch_size` points, it first samples a larger pool:

```python
X_pool = thompson_sampling(X_cand, num_samples=pool_size)
```

For example, `pool_size = 5 * batch_size`.

Then, a diversity filter selects the final batch by keeping points that are at least `min_dist` away from the points already chosen:

```python
X_next = diversity_filter(X_pool, batch_size, min_dist=0.05)
```

Only `X_next` is evaluated.

The goal is to reduce redundancy in batch Bayesian optimization. Standard batch Thompson sampling can choose points that are very close together, which may waste parallel evaluations. By oversampling first and filtering for distance, the batch is encouraged to cover more promising regions.

This method is not meant to be faster. It may be slightly slower because it samples more points. The intended benefit is better sample efficiency from more diverse batches.

