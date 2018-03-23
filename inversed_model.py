import chainer

import models


class InversedProbabilityDistributionNetwork(chainer.ChainList):
    def __init__(self, pdn, output_size, monotone_size, marginal_size, pool_size):
        super().__init__()
        self.pdn = pdn  # Don't register as a link. We don't want to train it here.
        for i in range(output_size):
            net = models.MonotonicNetwork([1] + monotone_size, [i] + marginal_size, pool_size)
            self.add_link(net)

    def sample(self, x):
        xp = self.xp
        p0 = xp.random.uniform(0.0, 1.0, x.shape).astype(x.dtype)
        xi_list = []
        for i, net in enumerate(self):
            p_i = p0[:, i:i + 1]
            if len(xi_list) == 0:
                marginal = xp.zeros((p_i.shape[0], 0), p_i.dtype)
            else:
                marginal = chainer.functions.concat(tuple(xi_list), 1)
            xi = net(p_i, marginal)
            xi_list.append(xi.data)
        x = xp.concatenate(tuple(xi_list), 1)
        return x

    def __call__(self, x):
        xp = self.xp
        p0 = xp.random.uniform(0.0, 1.0, x.shape).astype(x.dtype)
        xi_list = []
        loss = []
        for i, net in enumerate(self):
            p_i = p0[:, i:i + 1]
            if len(xi_list) == 0:
                marginal = xp.zeros((p_i.shape[0], 0), p_i.dtype)
            else:
                marginal = xp.concatenate(tuple(xi_list), 1)
            xi = net(p_i, marginal)
            yi = self.pdn[i](xi, marginal)
            loss_i = chainer.functions.bernoulli_nll(p_i, yi)
            loss_i = chainer.functions.sum(loss_i) / loss_i.size
            xi_list.append(xi.data)
            loss.append(loss_i)
        loss = chainer.functions.stack(tuple(loss))
        loss = chainer.functions.sum(loss)
        chainer.report({'loss': loss}, self)
        return loss
