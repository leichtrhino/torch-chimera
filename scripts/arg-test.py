#!/usr/bin/env python

import os
import sys

import torch
import torchaudio

from argparse import ArgumentParser
from itertools import chain

'''
Command line arguments
'''
def add_training_io_argument(parser):
    parser.add_argument('--train-dir', nargs='+', required=True, help='directory of training dataset')
    parser.add_argument('--validation-dir', nargs='*', help="directory of validation dataset")
    parser.add_argument('--output-model', help='output model file')
    parser.add_argument('--output-checkpoint', help='output checkout file')
    parser.add_argument('--input-model', help='input model file')
    parser.add_argument('--input-checkpoint', help='input checkpoint file')
    parser.add_argument('--sync', action='store_true', help='the dataset is synchronized (e.g. music and singing voice separation)')
    parser.add_argument('--log-file', help='log file')
    return parser

def validate_training_io_argument(args, parser):
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
    return args

def add_prediction_io_argument(parser):
    parser.add_argument('--data-dir', required=True, help='directory of dataset')
    parser.add_argument('--input-model', help='input model file')
    parser.add_argument('--input-checkpoint', help='input checkpoint file')
    parser.add_argument('--sync', action='store_true', help='the dataset is synchronized (e.g. music and singing voice separation)')
    parser.add_argument('--log-file', help='log file')
    return parser

def validate_prediction_io_argument(args, parser):
    if not os.path.isdir(args.data_dir):
        parser.error(f'"{args.data_dir}" is not a directory')
    if args.input_model and args.input_checkpoint:
        parser.error('passing both --input-model and --input-checkpoint is prohibited')
    if args.input_model and not os.path.isfile(args.input_model):
        parser.error(f'input model "{args.input_model}" is not a file')
    if args.input_checkpoint and not os.path.isfile(args.input_checkpoint):
        parser.error(f'input checkpoint "{args.input_model}" is not a file')
    return args

def add_evaluation_io_argument(parser):
    parser.add_argument('--data-dir', nargs='+', required=True, help="directory of validation dataset")
    parser.add_argument('--input-model', help='input model file')
    parser.add_argument('--input-checkpoint', help='input checkpoint file')
    parser.add_argument('--log-file', help='log file')
    return parser

def validate_evaluation_io_argument(args, parser):
    for d in args.data_dir:
        if not os.path.isdir(d):
            parser.error(f'"{d}" is not a directory')
    if args.input_model and args.input_checkpoint:
        parser.error('passing both --input-model and --input-checkpoint is prohibited')
    if args.input_model and not os.path.isfile(args.input_model):
        parser.error(f'input model "{args.input_model}" is not a file')
    if args.input_checkpoint and not os.path.isfile(args.input_checkpoint):
        parser.error(f'input checkpoint "{args.input_model}" is not a file')
    return args

def add_feature_argument(parser):
    parser.add_argument('--sr', type=int, default=8000, help='sampling rate')
    parser.add_argument('--n-fft', type=int, default=256, help='num of fft point')
    parser.add_argument('--segment-duration', type=float, default=3.0, help='segment duration in seconds')
    return parser

def validate_feature_argument(args, parser):
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
    return args

def add_model_argument(parser):
    parser.add_argument('--n-hidden', type=int, default=600, help='num of hidden state')
    parser.add_argument('--embedding-dim', type=int, default=20, help='embedding dimension of deep clustering')
    parser.add_argument('--residual', action='store_true', help='residual base module')
    return parser

def validate_model_argument(args, parser):
    if args.n_hidden <= 0:
        parser.error('--n-hidden is positive')
    if args.embedding_dim <= 0:
        parser.error('--embedding-dim is positive')
    return args

def add_training_argument(parser):
    parser.add_argument('--epoch', type=int, default=1, help='training epoch')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, help='learning rate. if not provided, 1e-3 (chimera++) or 1e-4 (wave-approximation) is set')
    parser.add_argument('--loss-function', required=True, choices=('chimera++', 'wave-approximation'))
    parser.add_argument('--permutation-free', action='store_true', help='enable permutation-free loss function')
    return parser

def validate_training_argument(args, parser):
    if args.epoch <= 0:
        parser.error('--epoch is positive')
    if args.lr is None:
        args.lr = 1e-3 if args.loss_function == 'chimera++' else\
            1e-4 if args.loss_function == 'wave-approximation' else\
            0
    if args.lr <= 0:
        parser.error('--lr is positive')
    return args

def parse_args():
    parser = ArgumentParser()
    subparser = parser.add_subparsers(help='<<subcommand help>>', dest='command')
    train_parser = subparser.add_parser('train', help='<<train help')
    train_io_parser = train_parser.add_argument_group('io')
    train_feature_parser = train_parser.add_argument_group('feature')
    train_model_parser = train_parser.add_argument_group('model')
    train_training_parser = train_parser.add_argument_group('training')
    add_training_io_argument(train_io_parser)
    add_feature_argument(train_feature_parser)
    add_model_argument(train_model_parser)
    add_training_argument(train_training_parser)

    predict_parser = subparser.add_parser('predict', help='<<predict help>>')
    predict_io_parser = predict_parser.add_argument_group('io')
    predict_feature_parser = predict_parser.add_argument_group('feature')
    predict_model_parser = predict_parser.add_argument_group('model')
    add_prediction_io_argument(predict_io_parser)
    add_feature_argument(predict_feature_parser)
    add_model_argument(predict_model_parser)

    evaluate_parser = subparser.add_parser('evaluate', help='<<evaluate help>>')
    evaluate_io_parser = evaluate_parser.add_argument_group('io')
    add_evaluation_io_argument(evaluate_io_parser)

    args = parser.parse_args()
    if args.command == 'train':
        validate_training_io_argument(args, parser)
        validate_feature_argument(args, parser)
        validate_model_argument(args, parser)
        validate_training_argument(args, parser)
    elif args.command == 'predict':
        validate_prediction_io_argument(args, parser)
        validate_feature_argument(args, parser)
        validate_model_argument(args, parser)
    elif args.command == 'evaluate':
        validate_evaluation_io_argument(args, parser)
    return args


def main():
    args = parse_args()
    print(args)

if __name__ == '__main__':
    main()
