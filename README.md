# Diversity Filtered Batch Thompson Sampling

This project implements a simple modification of batch Thompson sampling for Bayesian optimization. The baseline follows the BoTorch Thompson sampling tutorial: at each optimization step, a Gaussian process surrogate model is fit to the observed data, a large Sobol candidate set is generated over the normalized input domain, and `MaxPosteriorSampling` is used to choose the next batch of evaluation points.

The modification is to oversample Thompson candidates before selecting the final batch. Instead of directly requesting `batch_size` points from Thompson sampling, the algorithm first requests a larger pool of candidates:

```python
X_pool = thompson_sampling(X_cand, num_samples=pool_size)
```

where `pool_size` is larger than the final batch size, for example `pool_size = 5 * batch_size`. These points are all promising because each one was selected as the maximizer of a sampled posterior function over the candidate set.

A diversity filter is then applied to this pool. The filter greedily builds the final batch by starting with the first Thompson sampled point and then accepting later points only if they are at least `min_dist` away from all points already selected. The final batch is:

```python
X_next = diversity_filter(X_pool, batch_size, min_dist=0.05)
```

Only `X_next` is evaluated by the objective function.

The goal of this method is to reduce spatial redundancy in batch Bayesian optimization. Vanilla batch Thompson sampling can select points that are technically distinct but very close together in the input space. When evaluations are run in parallel, this can waste part of the batch exploring nearly the same region multiple times. By oversampling first and then selecting a more spread-out subset, the method encourages the batch to cover multiple promising regions while still using Thompson sampling to focus on high-value areas.

This approach is not intended to make batch generation faster. It may be slightly slower because it draws more Thompson samples. The intended benefit is improved sample efficiency through more diverse parallel evaluations.
