# Learning Cumulative Distribution Function with Neural Networks 
Here is a implementation of the new technique for density estimation with neural networks.

This model Learns cumulative distribution function with a *[Monotonic Networks (J. Sill, NIPS1997)](https://papers.nips.cc/paper/1358-monotonic-networks)*.

## 1d Gaussian
![Output heat map for 1d gaussian](images/gaussian_1d.png?raw=true)

## Truncated 1d Gaussian
![Output heat map for 1d gaussian](images/half_gaussian_1d.png?raw=true)

## Mixture of 2d Gaussians
![Output heat map for mixture of 2d gaussian](images/mixed_gaussian_2d.png?raw=true)

## Truncated 2d Gaussians
![Output heat map for truncated 2d gaussian](images/half_gaussian_2d.png?raw=true)

## Run
```
train.py --gpu 0 --dataset gaussian_1d
```
```
train.py --gpu 0 --dataset gaussian_mix_2d
```

# Sampling
You can also use this technique to sampling from the distribution by learning inverse of the model.

## 1d Mixed Gaussian
```
train.py --gpu 0 --dataset gaussian_mix_1d --resume result/forward_snapshot_epoch_100 --mode inverse
```

![Sampled data for mixture of 1d gaussian](images/mixed_gaussian_1d_sampled.png)

Learned distribution(orange line), sampled data from learned model(blue bars).

## 2d Mixed Gaussian
```
train.py --gpu 0 --dataset gaussian_mix_2d --resume result/forward_snapshot_epoch_100 --mode inverse
```

![Sampled data for mixture of 1d gaussian](images/mixed_gaussian_2d_sampled.png)

Original data(left), learned distribution(center), sampled data from learned model(right).
