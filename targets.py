import math


def gaussian_1d(xp, batchsize):
    return xp.random.normal(0.5, 0.25, (batchsize, 1)).astype(xp.float32)


def half_gaussian_1d(xp, batchsize):
    x = xp.random.normal(0, 0.25, (batchsize, 1)).astype(xp.float32)
    x = xp.fabs(x) + 0.25
    return x


def gaussian_2d(xp, batchsize):
    x = xp.random.normal(0.5, 0.25, (batchsize, 1)).astype(xp.float32)
    y = xp.random.normal(0.5, 0.25, x.shape).astype(xp.float32)
    return xp.concatenate((x, y), 1)


def gaussian_mixture_circle0(xp, batchsize):
    num_cluster = 8
    scale = 0.25
    var = 0.0625

    i = xp.random.randint(0, num_cluster, (batchsize, 1))
    radian = 2 * math.pi * i / num_cluster
    mean = xp.concatenate((xp.cos(radian), xp.sin(radian)), 1)
    var = xp.array(var).reshape(-1, 1)
    var = xp.broadcast_to(var, mean.shape)
    return (mean * scale + var * xp.random.normal(size=mean.shape)).astype(xp.float32)


def gaussian_mixture_circle(xp, batchsize):
    return gaussian_mixture_circle0(xp, batchsize) + 0.5


def truncated_gaussian_circle(xp, batchsize):
    x = xp.random.normal(0.0, 0.25, (batchsize, 1)).astype(xp.float32)
    y = xp.random.normal(0.0, 0.25, x.shape).astype(xp.float32)
    samples = xp.concatenate((x, y), 1)

    mask = (samples[:, 0] > 0) == (samples[:, 1] > 0)
    samples[:, 0] = samples[:, 0] * (1 - mask * 2)
    return samples + 0.5


def gaussian_mixture_circle_1d(xp, batchsize):
    samples = gaussian_mixture_circle(xp, batchsize)
    return samples[:, 0:1]
