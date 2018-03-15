import pathlib

import chainer
import chainer.backends.cuda

import seaborn
from matplotlib import pyplot


class Visualize(chainer.training.Extension):
    trigger = (1, 'epoch')

    def __init__(self, target, samples):
        self.target = target
        self.samples = samples

    def __call__(self, trainer):
        path = pathlib.Path(trainer.out) / pathlib.Path("epoch_{.updater.epoch}.png".format(trainer))
        with chainer.using_config('train', False):
            if self.samples.shape[1] == 1:
                self.save_1d(self.target, self.samples, str(path))
            else:
                self.save_2d(self.target, self.samples, str(path))

    @staticmethod
    def save_1d(model, samples, file_path):
        seaborn.set_style("whitegrid")

        xp = model.xp
        x = xp.linspace(0.0, 1.0, 256).astype(xp.float32)
        p = model.calculate_p(x[:, None])

        x = chainer.backends.cuda.to_cpu(x)
        p = chainer.backends.cuda.to_cpu(p.data)
        samples = chainer.backends.cuda.to_cpu(samples)

        pyplot.figure()
        ax = seaborn.distplot(samples, kde=False, bins=64)
        pyplot.plot(x, p * samples.size * ax.patches[0].get_width())
        pyplot.savefig(file_path)
        pyplot.close()

    @staticmethod
    def save_2d(model, samples, file_path):
        seaborn.set_style("whitegrid")

        xp = model.xp

        x = xp.linspace(0.0, 1.0, 65, dtype=xp.float32)
        x1, x2 = xp.broadcast_arrays(x[:, None], x[None])
        x = xp.stack((x1, x2), 2)
        x = xp.reshape(x, (-1, 2))
        p = model.calculate_p(x)
        p = chainer.functions.reshape(p, x1.shape)
        p = chainer.backends.cuda.to_cpu(p.data)

        pyplot.figure()

        ax1 = pyplot.subplot(1, 2, 1)
        ax1.set_aspect('equal')
        pyplot.scatter(samples[:, 0], samples[:, 1], 1)

        ax2 = pyplot.subplot(1, 2, 2)
        ax2.set_aspect('equal')
        pyplot.pcolor(p)

        pyplot.savefig(file_path, bbox_inches='tight')
        pyplot.close()
