from __future__ import print_function

import argparse

import chainer
import numpy
from chainer import training
from chainer.backends.cuda import get_device_from_id
from chainer.training import extensions

import inversed_model
import made
import models
import targets
import visualize


def main():
    parser = argparse.ArgumentParser(description='Learning cumulative distribution function with Monotonic Networks.')
    parser.add_argument('--dataset', '-d', default='gaussian_1d',
                        help='The dataset to use: gaussian_1d or gaussian_mix_2d')
    parser.add_argument('--mode', '-m', default='forward',
                        help='Learning probability distribution(\'forward\') or it\'s inverse function(\'inverse\')')
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    if args.dataset == 'gaussian_1d':
        train = targets.gaussian_1d(numpy, 16384)
        test = targets.gaussian_1d(numpy, 1024)
    elif args.dataset == 'gaussian_half_1d':
        train = targets.half_gaussian_1d(numpy, 16384)
        test = targets.half_gaussian_1d(numpy, 1024)
    elif args.dataset == 'gaussian_mix_1d':
        train = targets.gaussian_mixture_1d(numpy, 32768)
        test = targets.gaussian_mixture_1d(numpy, 1024)
    elif args.dataset == 'mixture_1d':
        train = targets.mixture_1d(numpy, 32768)
        test = targets.mixture_1d(numpy, 1024)
    elif args.dataset == 'gaussian_mix_2d':
        train = targets.gaussian_mixture_circle(numpy, 32768)
        test = targets.gaussian_mixture_circle(numpy, 1024)
    elif args.dataset == 'half_gaussian_2d':
        train = targets.truncated_gaussian_circle(numpy, 32768)
        test = targets.truncated_gaussian_circle(numpy, 1024)
    else:
        raise RuntimeError('Invalid dataset: {}.'.format(args.dataset))

    if train.shape[1] == 1:
        model = models.ProbabilityDistributionNetwork(1, [64, 128, 64], [32, 32], 8)
    elif train.shape[1] == 2:
        # model = models.ProbabilityDistributionNetwork(2, [32, 32, 32], [32, 32], 8)
        model = made.MaskedMonotonicNetwork([2, 32, 32], 0, 4, 4)
    else:
        raise RuntimeError('Invalid dataset.')

    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize, False, False)

    stop_trigger = (args.epoch, 'epoch')

    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, stop_trigger, out=args.out)
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
    trainer.extend(chainer.training.extensions.FailOnNonNumber())

    trainer.extend(extensions.snapshot(filename='forward_snapshot_epoch_{.updater.epoch}'), trigger=(50, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'elapsed_time']))
    trainer.extend(visualize.Visualize("forward_{.updater.epoch}.png", model, test, None), trigger=(10, 'epoch'))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    if args.mode == 'forward':
        trainer.run()

    train_inverse(args, train, test, model)


def train_inverse(args, train, test, pdn):
    if train.shape[1] == 1:
        model = inversed_model.InversedProbabilityDistributionNetwork(pdn, 1)
    elif train.shape[1] == 2:
        model = inversed_model.InversedProbabilityDistributionNetwork(pdn, 2)
    else:
        raise RuntimeError('Invalid dataset.')

    if args.gpu >= 0:
        model.to_gpu()

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize, False, False)

    stop_trigger = (args.epoch, 'epoch')

    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, stop_trigger, out=args.out)
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

    trainer.extend(extensions.snapshot(filename='inverse_snapshot_epoch_{.updater.epoch}'), trigger=(50, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'elapsed_time']))
    trainer.extend(visualize.Visualize("inverse_{.updater.epoch}.png", pdn, test, model), trigger=(10, 'epoch'))

    trainer.run()


if __name__ == '__main__':
    main()
