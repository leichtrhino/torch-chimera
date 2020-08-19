#!/usr/bin/env python

import os
import sys

import torch
import torchaudio

from argparse import ArgumentParser
from itertools import chain

def parse_args():
    parser = ArgumentParser()
    # general options
    parser.add_argument('-v', '--verbose', action='store_true')

    # input and output
    group_io = parser.add_argument_group('input and output')
    group_io.add_argument('--train-dir', nargs='+', required=True, help='directory of training dataset')
    group_io.add_argument('--validation-dir', nargs='*', help="directory of validation dataset")
    group_io.add_argument('--output-model', help='output model file')
    group_io.add_argument('--output-checkpoint', help='output checkout file')
    group_io.add_argument('--input-model', help='input model file')
    group_io.add_argument('--input-checkpoint', help='input checkpoint file')
    group_io.add_argument('--log-file', help='log file')

    # feature param. (e.g. sampling rate, dft point, segment duration)
    group_feat = parser.add_argument_group('feature')
    group_feat.add_argument('--sr', type=int, default=8000, help='sampling rat')
    group_feat.add_argument('--n-fft', type=int, default=256, help='num of fft point')
    group_feat.add_argument('--segment-duration', type=float, default=3.0, help='segment duration in seconds')

    # model param.
    group_model = parser.add_argument_group('model')
    group_model.add_argument('--n-hidden', type=int, default=600, help='num of hidden state')
    group_model.add_argument('--embedding-dim', type=int, default=20, help='embedding dimension of deep clustering')
    group_model.add_argument('--residual', action='store_true', help='residual base module')

    # training param.
    group_train = parser.add_argument_group('training')
    group_train.add_argument('--epoch', type=int, default=1, help='training epoch')
    group_train.add_argument('--batch-size', type=int, default=32, help='batch size')
    group_train.add_argument('--lr', type=float, help='learning rate. if not provided, 1e-3 (chimera++) or 1e-4 (wave-approximation) is set')
    group_train.add_argument('--loss-function', required=True, choices=('chimera++', 'wave-approximation'))
    group_train.add_argument('--sync', action='store_true', help='the dataset is synchronized (e.g. music and singing voice separation)')
    group_train.add_argument('--permutation-free', action='store_true', help='enable permutation-free loss function')

    args = parser.parse_args()

    # validation and post processing
    # input and output
    if args.validation_dir and len(args.validation_dir) != len(args.train_dir):
        parser.error('the number of train and validation dir mismatch')
    for d in chain(
            args.train_dir,
            args.validation_dir if args.validation_dir else tuple()
    ):
        if not os.path.isdir(d):
            parser.error(f'"{d}" is not a directory')
    if args.input_model and args.input_checkpoint:
        parser.error('passing both --input-model and --input-checkpoint is prohibited')
    if args.input_model and not os.path.isfile(args.input_model):
        parser.error(f'input model "{args.input_model}" is not a file')
    if args.input_checkpoint and not os.path.isfile(args.input_checkpoint):
        parser.error(f'input checkpoint "{args.input_model}" is not a file')
    if not args.output_model and not args.output_checkpoint\
       or args.output_model and args.output_checkpoint:
        parser.error('either --output-model or --output-checkpoint must be provided')

    # feature parameters
    if args.sr <= 0:
        parser.error('--sr is positive')
    if args.n_fft  <= 0:
        parser.error('--n-fft is positive')
    if args.segment_duration <= 0:
        parser.error('--segment-duration is positive')
    args.win_length = args.n_fft
    args.hop_length = args.win_length // 4;
    args.bin_num, args.frame_num, _ = torch.stft(
        torch.Tensor(int(args.segment_duration * args.sr)),
        args.n_fft,
        args.hop_length,
        args.win_length
    ).shape
    (args.segment_length,) = torchaudio.functional.istft(
        torch.Tensor(args.bin_num, args.frame_num, 2),
        args.n_fft,
        args.hop_length,
        args.win_length
    ).shape

    # model parameters
    if args.n_hidden <= 0:
        parser.error('--n-hidden is positive')
    if args.embedding_dim <= 0:
        parser.error('--embedding-dim is positive')

    # training param
    if args.epoch <= 0:
        parser.error('--epoch is positive')
    if args.lr is None:
        args.lr = 1e-3 if args.loss_function == 'chimera++' else\
            1e-4 if args.loss_function == 'wave-approximation' else\
            0
    if args.lr <= 0:
        parser.error('--lr is positive')

    return args

def main():
    args = parse_args()
    print(args)

if __name__ == '__main__':
    main()
