## Lookahead Optimizer
This is a tensorflow implementation of lookahead optimizer. Read the [paper](https://arxiv.org/abs/1907.08610)
for
details.

It is essentially a wrapper around any optimizer that looks ahead k steps and
linearly interploates other weights accordingly to that direction.

Tested with tensorflow 1.13

## Example

```python
fast_optimizer = optimizer.AdamOptimizer(0.001)
train_step = lookahead.LookaheadOptimizer(fast_optimizer).minimize(loss)

```

## Disclaimer

It suffers from conditional expression to check current step has reached k, and will cause a branch divergence in GPU warps. The wall clock time on GPU has increased by 290%. Looking for contributions to remove branch expression.

I am working on contributing to tensorflow repository using cuda API to optimize its evaluation. If you have better idea, feel free to post an issue.

## Other implementations

https://github.com/Janus-Shiau/lookahead_tensorflow
