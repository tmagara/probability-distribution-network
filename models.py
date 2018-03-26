import chainer


class MonotonicLinear(chainer.Chain):
    """Linear layer with a constraint: weight > 0"""

    def __init__(self, monotone_input, marginal_input, monotone_output, marginal_output):
        super().__init__()
        with self.init_scope():
            weight_initializer = chainer.initializers.LeCunNormal()
            self.w = chainer.Parameter(weight_initializer, (monotone_output, monotone_input))
            bias_initializer = chainer.initializers.Constant(0)
            self.b1 = chainer.Parameter(bias_initializer, (monotone_output,))
            if marginal_output == 0:
                self.b2 = None
            else:
                self.b2 = chainer.Parameter(bias_initializer, (marginal_output,))

            if marginal_input == 0:
                self.linear1 = None
            else:
                self.linear1 = chainer.links.Linear(marginal_input, monotone_output, True)

            if marginal_input == 0 or marginal_output == 0:
                self.linear2 = None
            else:
                self.linear2 = chainer.links.Linear(marginal_input, marginal_output, True)

    def __call__(self, monotone, marginal):
        xp = self.xp

        absolute_w = chainer.functions.absolute(self.w)
        h1 = chainer.functions.linear(monotone, absolute_w, self.b1)

        if self.linear1 is not None:
            h1 += self.linear1(marginal)

        if self.b2 is None:
            h2 = xp.zeros((h1.shape[0], 0), h1.dtype)
        else:
            h2 = self.b2[None]
            h2 = chainer.functions.broadcast_to(h2, (h1.shape[0],) + h2.shape[1:])

        if self.linear2 is not None:
            h2 += self.linear2(marginal)
        return h1, h2


class MonotonicNetwork(chainer.Chain):
    """Multi-layer fully connected network which represents monotonic function.

    See:
        `Monotonic Networks <https://papers.nips.cc/paper/1358-monotonic-networks>`_ [J. Sill, NIPS1997].

    """

    def __init__(self, monotone_size, marginal_size, pool_size):
        super().__init__()
        with self.init_scope():
            self.linear1 = MonotonicLinear(monotone_size[0], marginal_size[0], monotone_size[1], marginal_size[1])
            self.linear2 = MonotonicLinear(monotone_size[1], marginal_size[1], monotone_size[2], marginal_size[2])
            self.linear3 = MonotonicLinear(monotone_size[2], marginal_size[2], monotone_size[3], 0)
        self.pool_size = pool_size

    def __call__(self, monotone, marginal):
        monotone, marginal = self.linear1(monotone, marginal)
        monotone = chainer.functions.tanh(monotone)
        marginal = chainer.functions.relu(marginal)

        monotone, marginal = self.linear2(monotone, marginal)
        monotone = chainer.functions.tanh(monotone)
        marginal = chainer.functions.relu(marginal)

        h, _ = self.linear3(monotone, marginal)

        h = chainer.functions.reshape(h, (h.shape[0], 1, self.pool_size, -1))
        h = chainer.functions.max(h, 3)
        h = chainer.functions.min(h, 2)

        return h


class ProbabilityDistributionNetwork(chainer.ChainList):
    """Probability distribution function with multi-dimensional inputs."""
    def __init__(self, input_size, monotone_size, marginal_size, pool_size):
        super().__init__()
        for i in range(input_size):
            net = MonotonicNetwork([1] + monotone_size, [i] + marginal_size, pool_size)
            self.add_link(net)

    def calculate_pn(self, x):
        xi_list = chainer.functions.separate(x[:, :, None], 1)
        yi_list = []
        for i, cumulative_distribution in enumerate(self):
            marginal = chainer.as_variable(x[:, 0:i].data)
            with chainer.using_config('enable_backprop', True):
                yi = cumulative_distribution(xi_list[i], marginal)
            yi_list.append(yi)
        gy_list = chainer.grad(yi_list, xi_list, enable_double_backprop=True)
        y = chainer.functions.concat(tuple(yi_list), 1)
        gy = chainer.functions.concat(tuple(gy_list), 1)
        return y, gy

    def calculate_p(self, x):
        x = chainer.Variable(x)
        y, gy = self.calculate_pn(x)
        cumulative = chainer.functions.sigmoid(y)
        p = cumulative * (1 - cumulative) * gy
        p = chainer.functions.prod(p, 1, True)
        return p

    def __call__(self, x):
        x = chainer.Variable(x)
        y, gy = self.calculate_pn(x)
        log_p = chainer.functions.softplus(y) * 2 - y - chainer.functions.log(gy)
        loss = chainer.functions.sum(log_p) / log_p.shape[0]
        chainer.report({'loss': loss}, self)
        return loss
