import pathlib

import chainer
import chainer.backends.cuda

import seaborn
from matplotlib import pyplot


class Visualize(chainer.training.Extension):
    trigger = (1, 'epoch')

    def __init__(self, path_string, target, samples, inverse):
        self.path_string = path_string
        self.target = target
        self.samples = samples
        self.inverse = inverse

    def __call__(self, trainer):
        path = pathlib.Path(trainer.out) / pathlib.Path(self.path_string.format(trainer))
        with chainer.using_config('train', False):
            if self.samples.shape[1] == 1:
                self.save_1d(str(path))
            else:
                self.save_2d(str(path))

    def save_1d(self, file_path):
        model = self.target
        samples = self.samples

        seaborn.set_style("whitegrid")

        xp = model.xp
        x = xp.linspace(-0.5, 0.5, 257).astype(xp.float32)
        p = model.calculate_p(x[:, None])

        x = chainer.backends.cuda.to_cpu(x)
        p = chainer.backends.cuda.to_cpu(p.data)
        samples = chainer.backends.cuda.to_cpu(samples)

        pyplot.figure()

        if self.inverse is not None:
            uniform = xp.random.uniform(size=samples.shape).astype(samples.dtype)
            generated = self.inverse.sample(uniform).data
        else:
            generated = samples
        generated = chainer.backends.cuda.to_cpu(generated)
        ax = seaborn.distplot(generated, kde=False, bins=64)

        pyplot.plot(x, p * samples.size * ax.patches[0].get_width())

        pyplot.savefig(file_path)
        pyplot.close()

    def save_2d(self, file_path):
        model = self.target
        samples = self.samples

        seaborn.set_style("whitegrid")

        xp = model.xp

        x = xp.linspace(-0.5, 0.5, 65, dtype=xp.float32)
        x1, x2 = xp.broadcast_arrays(x[:, None], x[None])
        x = xp.stack((x1, x2), 2)
        x = xp.reshape(x, (-1, 2))

        p = model.calculate_p(x)
        p = chainer.functions.prod(p, 1, True)
        p = chainer.functions.reshape(p, x1.shape)
        p = chainer.backends.cuda.to_cpu(p.data)

        pyplot.figure()

        rows = 2 if self.inverse is None else 3

        ax1 = pyplot.subplot(1, rows, 1)
        ax1.set_aspect('equal')
        pyplot.scatter(samples[:, 0], samples[:, 1], 1)

        ax2 = pyplot.subplot(1, rows, 2)
        ax2.set_aspect('equal')
        pyplot.pcolor(p)

        if self.inverse is not None:
            uniform = xp.random.uniform(size=samples.shape, dtype=samples.dtype)
            generated = self.inverse.sample(uniform)
            generated = chainer.backends.cuda.to_cpu(generated)

            ax3 = pyplot.subplot(1, rows, 3)
            ax3.set_aspect('equal')
            pyplot.scatter(generated[:, 0], generated[:, 1], 1)

        pyplot.savefig(file_path, bbox_inches='tight')
        pyplot.close()
