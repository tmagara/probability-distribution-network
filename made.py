import chainer
import numpy


class MaskedMonotonicLinear(chainer.Chain):

    def __init__(self, input_count, output_count):
        super().__init__()
        with self.init_scope():
            weight_initializer = chainer.initializers.LeCunNormal()
            self.w = chainer.Parameter(weight_initializer, (output_count, input_count))
            bias_initializer = chainer.initializers.Constant(0)
            self.b = chainer.Parameter(bias_initializer, (output_count,))

        self.dependency_mask = numpy.zeros((output_count, input_count), numpy.bool)
        self.register_persistent('dependency_mask')
        self.monotone_mask = numpy.zeros((output_count, input_count), numpy.bool)
        self.register_persistent('monotone_mask')

    def mask(self, dependency_in, dependency_out, mask_in, mask_out):
        xp = self.xp

        d_in, d_out = xp.broadcast_arrays(dependency_in[None, :], dependency_out[:, None])
        m_in, m_out = xp.broadcast_arrays(mask_in[None, :], mask_out[:, None])

        self.dependency_mask[:] = ((d_in < d_out) & ~m_in) | ((d_in == d_out) & (m_in == m_out))
        self.monotone_mask[:] = m_in

    def __call__(self, x):
        sign = ((self.w.data < 0) & self.monotone_mask) * -2 + 1
        w = (sign * self.dependency_mask) * self.w
        return chainer.functions.linear(x, w, self.b)


class MaskedMonotonicNetwork(chainer.Chain):

    def __init__(self, unit_counts, marginal_count, pool_size1, pool_size2):
        super().__init__()

        xp = numpy

        marginal1_indices = xp.zeros(marginal_count, xp.int32)
        marginal2_indices = xp.arange(unit_counts[0] - marginal_count - 1) + 1
        monotone_indices = xp.arange(unit_counts[0] - marginal_count) + 1
        d0 = xp.concatenate((marginal1_indices, marginal2_indices, monotone_indices))

        marginal1_masks = xp.zeros(marginal_count, xp.bool)
        marginal2_masks = xp.zeros(unit_counts[0] - marginal_count - 1, xp.bool)
        monotone_masks = xp.ones(unit_counts[0] - marginal_count, xp.bool)
        m0 = xp.concatenate((marginal1_masks, marginal2_masks, monotone_masks))

        pool_indices = xp.tile(monotone_indices, pool_size1 * pool_size2)
        pool_masks = xp.tile(monotone_masks, pool_size1 * pool_size2)

        self.dependency = [d0] + [xp.zeros(c, xp.int32) for c in unit_counts[1:]] + [pool_indices]
        self.monotone_mask = [m0] + [xp.zeros(c, xp.bool) for c in unit_counts[1:]] + [pool_masks]
        self.register_persistent('dependency')

        self.linears = [MaskedMonotonicLinear(d0.shape[0], unit_counts[1]),
                        MaskedMonotonicLinear(unit_counts[1], unit_counts[2]),
                        MaskedMonotonicLinear(unit_counts[2], pool_indices.shape[0])]
        with self.init_scope():
            self.linear1 = self.linears[0]
            self.linear2 = self.linears[1]
            self.linear3 = self.linears[2]

        self.pool_size1 = pool_size1
        self.pool_size2 = pool_size2

        self.shuffle(0xABCDEF)

    def shuffle(self, seed):
        xp = self.xp
        xp.random.seed(seed)

        base_dependency = self.dependency[0]
        base_mask = self.monotone_mask[0]
        base_count = base_dependency.shape[0]
        base_indices = xp.arange(base_count)
        for d, m in zip(self.dependency[1:-1], self.monotone_mask[1:-1]):
            count = d.shape[0]
            indices = xp.repeat(base_indices, ((count - 1) // base_count) + 1)
            indices = xp.random.permutation(indices)[:count]
            d[:] = base_dependency[indices]
            m[:] = base_mask[indices]

        for l, d1, d2, m1, m2 in zip(
                self.linears,
                self.dependency[:-1],
                self.dependency[1:],
                self.monotone_mask[:-1],
                self.monotone_mask[1:]):
            l.mask(d1, d2, m1, m2)

    def process(self, x):
        h = self.linear1(x)
        h = chainer.functions.tanh(h)

        h = self.linear2(h)
        h = chainer.functions.tanh(h)

        h = self.linear3(h)

        h = chainer.functions.reshape(h, (h.shape[0], self.pool_size1, self.pool_size2, -1))
        h = chainer.functions.max(h, 2)
        h = chainer.functions.min(h, 1)

        return h

    def cumulative_p(self, x):
        x = chainer.functions.concat((x[:, :-1], x))
        return self.process(x)

    def calculate_p(self, x):
        monotone = chainer.Variable(x, requires_grad=True)
        marginal = chainer.Variable(x[:, :-1], requires_grad=False)
        with chainer.using_config('enable_backprop', True):
            x = chainer.functions.concat((marginal, monotone))
            y = self.process(x)
            p_cumulative = chainer.functions.sigmoid(y)
        p_list = chainer.grad([p_cumulative], [monotone], enable_double_backprop=True)
        return p_list[0]

    def __call__(self, x):
        p = self.calculate_p(x)
        loss = -chainer.functions.log(p)
        loss = chainer.functions.sum(loss) / loss.shape[0]
        chainer.report({'loss': loss}, self)
        return loss
