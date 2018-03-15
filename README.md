# Probability Distribution Network
Learning cumulative distribution function with a *[Monotonic Networks (J. Sill, NIPS1997)](https://papers.nips.cc/paper/1358-monotonic-networks) .*

## 1d Gaussian
![Output heat map for 1d gaussian](images/gaussian_1d.png?raw=true)

## Truncated 1d Gaussian
![Output heat map for 1d gaussian](images/half_gaussian_1d.png?raw=true)

## Mixture of 2d Gaussians
![Output heat map for mixture of 2d gaussian](images/mixed_gaussian_2d.png?raw=true)

## Truncated 2d Gaussians
![Output heat map for truncated 2d gaussian](images/half_gaussian_2d.png?raw=true)

##Run
```
train.py --gpu 0 --dataset gaussian_1d
```
```
train.py --gpu 0 --dataset gaussian_mix_2d
```
