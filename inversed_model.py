import chainer


class InversedProbabilityDistributionNetwork(chainer.Chain):
    def __init__(self, pdn, output_size):
        super().__init__()
        self.pdn = pdn  # Don't register as a link. We don't want to train it here.
        units = [output_size, 64, 128, 64, output_size]
        initializer = chainer.initializers.HeNormal()
        with self.init_scope():
            self.linear0 = chainer.links.Linear(units[0], units[1], initialW=initializer)
            self.linear1 = chainer.links.Linear(units[1], units[2], initialW=initializer)
            self.linear2 = chainer.links.Linear(units[2], units[3], initialW=initializer)
            self.linear3 = chainer.links.Linear(units[3], units[4], initialW=initializer)

    def sample(self, p0):
        h = p0

        h = self.linear0(h)
        h = chainer.functions.relu(h)
        h = self.linear1(h)
        h = chainer.functions.relu(h)
        h = self.linear2(h)
        h = chainer.functions.relu(h)
        h = self.linear3(h)

        return h

    def log_mse(self, x0, x1, axis):
        dd = chainer.functions.square(x0 - x1)
        mse = chainer.functions.mean(dd, axis, keepdims=True)
        log_mse = 0.5 * chainer.functions.log(mse)
        return log_mse

    def __call__(self, x0):
        xp = self.xp
        p0 = xp.random.uniform(0.0, 1.0, x0.shape).astype(x0.dtype)

        x = self.sample(p0)
        y = self.pdn.cumulative_p(x)
        p = chainer.functions.sigmoid(y)

        # p_loss = chainer.functions.bernoulli_nll(p0, y) / y.size
        p_loss = self.log_mse(p0, p, (0, ))
        p_loss = chainer.functions.sum(p_loss)

        chainer.report({'loss': p_loss}, self)
        return p_loss
