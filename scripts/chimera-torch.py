#!/usr/bin/env python

import os
import sys
import random

import torch
import torchaudio
import resampy

from argparse import ArgumentParser
from itertools import chain

try:
    import torchchimera
except:
    # attempts to import local module
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    import torchchimera
from torchchimera.datasets import FolderTuple
from torchchimera.models import ChimeraMagPhasebook
from torchchimera.losses import permutation_free
from torchchimera.losses import loss_mi_tpsa
from torchchimera.losses import loss_dc_deep_lda
from torchchimera.losses import loss_wa
from torchchimera.losses import loss_csa
from torchchimera.metrics import eval_snr
from torchchimera.metrics import eval_si_sdr

'''
Datasets, transforms and their utility
'''
class Stft(torch.nn.Module):
    def __init__(self, n_fft, hop_length=None, win_length=None, window=None):
        super(Stft, self).__init__()
        self.n_fft = n_fft
        if hop_length is None:
            self.hop_length = n_fft // 4
        else:
            self.hop_length = hop_length
        if win_length is None:
            self.win_length = n_fft
        else:
            self.win_length = win_length
        self.window = window
    def forward(self, x):
        waveform_length = x.shape[-1]
        y = torch.stft(
            x.reshape(x.shape[:-1].numel(), waveform_length),
            self.n_fft,
            self.hop_length,
            self.win_length,
            window=self.window
        )
        _, freq, time, _ = y.shape
        return y.reshape(*x.shape[:-1], freq, time, 2)

class Istft(torch.nn.Module):
    def __init__(self, n_fft, hop_length=None, win_length=None, window=None):
        super(Istft, self).__init__()
        self.n_fft = n_fft
        if hop_length is None:
            self.hop_length = n_fft // 4
        else:
            self.hop_length = hop_length
        if win_length is None:
            self.win_length = n_fft
        else:
            self.win_length = win_length
        self.window = window
    def forward(self, x):
        freq, time = x.shape[-3], x.shape[-2]
        y = torchaudio.functional.istft(
            x.reshape(x.shape[:-3].numel(), freq, time, 2),
            self.n_fft,
            self.hop_length,
            self.win_length,
            window=self.window
        )
        waveform_length = y.shape[-1]
        return y.reshape(*x.shape[:-3], waveform_length)

def dc_label_matrix(S):
    batch_size, n_channels, freq_bins, spec_time, _ = S.shape
    S_abs = S.norm(p=2, dim=-1)
    p = S_abs.transpose(1, 3).reshape(batch_size, spec_time*freq_bins, n_channels).softmax(dim=-1).cumsum(dim=-1)
    r = torch.rand(batch_size, spec_time * freq_bins, device=S.device)
    k = torch.eye(n_channels, device=S.device)[torch.argmin(torch.where(r.unsqueeze(-1) <= p, p, torch.ones_like(p)), dim=-1)]
    return k

def dc_weight_matrix(X):
    batch_size, freq_bins, spec_time, _ = X.shape
    X_abs = X.norm(p=2, dim=-1)
    weight = X_abs.transpose(1, 2).reshape(batch_size, spec_time*freq_bins)\
        / X_abs.sum(dim=(1, 2)).clamp(min=1e-16).unsqueeze(-1)
    return weight

def comp_mul(X, Y):
    (X_re, X_im), (Y_re, Y_im) = X.unbind(-1), Y.unbind(-1)
    return torch.stack((
        X_re * Y_re - X_im * Y_im,
        X_re * Y_im + X_im * Y_re
    ), dim=-1)

def make_x_in(batch, args):
    window = torch.sqrt(torch.hann_window(args.n_fft, device=batch.device))
    stft = Stft(args.n_fft, args.hop_length, args.win_length, window)
    s, x = batch, batch.sum(dim=1)
    S, X = stft(s), stft(x)
    return s, x, S, X

def forward(model, x_in, args):
    s, x, S, X = x_in
    embd, (mag, com), _ = model(
        torch.log10(X.norm(p=2, dim=-1).clamp(min=1e-40)),
        outputs=['mag', 'com']
    )
    return embd, (mag, com)

def compute_loss(x_in, y_pred, args):
    window = torch.sqrt(torch.hann_window(args.n_fft, device=x_in[0].device))
    istft = Istft(args.n_fft, args.hop_length, args.win_length, window)
    s, x, S, X = x_in
    embd, (mag, com) = y_pred
    if args.loss_function == 'chimera++':
        Y = dc_label_matrix(S)
        weight = dc_weight_matrix(X)
        alpha = 0.975
        loss_dc = alpha * loss_dc_deep_lda(embd, Y, weight)
        if args.permutation_free:
            loss_mi = (1-alpha) * permutation_free(loss_mi_tpsa)(mag, X, S, gamma=2.)
        else:
            loss_mi = (1-alpha) * loss_mi_tpsa(mag, X, S, gamma=2.)
        loss = loss_dc + loss_mi
    elif args.loss_function == 'wave-approximation':
        Shat = comp_mul(com, X.unsqueeze(1))
        shat = istft(Shat)
        waveform_length = min(s.shape[-1], shat.shape[-1])
        s = s[:, :, :waveform_length]
        shat = shat[:, :, :waveform_length]
        if args.permutation_free:
            loss = permutation_free(loss_wa)(shat, s)
        else:
            loss = loss_wa(shat, s)
    return loss

def predict_waveform(model, mixture, args):
    window = torch.sqrt(torch.hann_window(args.n_fft)).to(args.device)
    stft = Stft(args.n_fft, args.hop_length, args.win_length, window)
    istft = Istft(args.n_fft, args.hop_length, args.win_length, window)
    X = stft(mixture)
    _, (com,), _ = model(
        torch.log10(X.norm(p=2, dim=-1).clamp(min=1e-40)),
        outputs=['com']
    )
    Shat = comp_mul(com, X.unsqueeze(1))
    return istft(Shat)

'''
Command line arguments
'''
def add_general_argument(parser):
    parser.add_argument('--gpu', action='store_true', help='enable cuda device')
    parser.add_argument('--segment-duration', type=float, help='segment duration in seconds')
    return parser

def validate_general_argument(args, parser):
    if args.gpu and not torch.cuda.is_available():
        parser.error(f'cuda is not available')
    # get prefered device
    args.device = torch.device('cuda' if args.gpu else 'cpu')
    if args.segment_duration is not None and args.segment_duration <= 0:
        parser.error('--segment-duration is positive')
    return args

def add_training_io_argument(parser):
    parser.add_argument('--train-dir', nargs='+', required=True, help='directory of training dataset')
    parser.add_argument('--validation-dir', nargs='*', help="directory of validation dataset")
    parser.add_argument('--output-model', help='output model file')
    parser.add_argument('--output-checkpoint', help='output checkout file')
    parser.add_argument('--input-model', help='input model file')
    parser.add_argument('--input-checkpoint', help='input checkpoint file')
    parser.add_argument('--sync', action='store_true', help='the dataset is synchronized (e.g. music and singing voice separation)')
    parser.add_argument('--permutation-free', action='store_true', help='enable permutation-free loss function')
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
    parser.add_argument('--input-file', required=True, help='input wav file')
    parser.add_argument('--output-files', required=True, nargs='+', help='output wav files')
    parser.add_argument('--input-model', help='input model file')
    parser.add_argument('--input-checkpoint', help='input checkpoint file')
    parser.add_argument('--log-file', help='log file')
    return parser

def validate_prediction_io_argument(args, parser):
    if not os.path.isfile(args.input_file):
        parser.error(f'"{args.input_file}" is not a file')
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
    parser.add_argument('--output-file', help='output file')
    parser.add_argument('--log-file', help='log file')
    parser.add_argument('--sync', action='store_true', help='the dataset is synchronized (e.g. music and singing voice separation)')
    parser.add_argument('--permutation-free', action='store_true', help='enable permutation-free evaluation function')
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
    return parser

def validate_feature_argument(args, parser):
    if args.sr <= 0:
        parser.error('--sr is positive')
    if args.n_fft  <= 0:
        parser.error('--n-fft is positive')
    args.win_length = args.n_fft
    args.hop_length = args.win_length // 4;
    args.bin_num = args.n_fft // 2 + 1
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
    return parser

def validate_training_argument(args, parser):
    if args.epoch < 0:
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
    train_general_parser = train_parser.add_argument_group('general')
    train_io_parser = train_parser.add_argument_group('io')
    train_feature_parser = train_parser.add_argument_group('feature')
    train_model_parser = train_parser.add_argument_group('model')
    train_training_parser = train_parser.add_argument_group('training')
    add_general_argument(train_general_parser)
    add_training_io_argument(train_io_parser)
    add_feature_argument(train_feature_parser)
    add_model_argument(train_model_parser)
    add_training_argument(train_training_parser)

    predict_parser = subparser.add_parser('predict', help='<<predict help>>')
    predict_general_parser = predict_parser.add_argument_group('general')
    predict_io_parser = predict_parser.add_argument_group('io')
    predict_feature_parser = predict_parser.add_argument_group('feature')
    predict_model_parser = predict_parser.add_argument_group('model')
    add_general_argument(predict_general_parser)
    add_prediction_io_argument(predict_io_parser)
    add_feature_argument(predict_feature_parser)
    add_model_argument(predict_model_parser)

    evaluate_parser = subparser.add_parser('evaluate', help='<<evaluate help>>')
    evaluate_general_parser = evaluate_parser.add_argument_group('general')
    evaluate_io_parser = evaluate_parser.add_argument_group('io')
    evaluate_feature_parser = evaluate_parser.add_argument_group('feature')
    evaluate_model_parser = evaluate_parser.add_argument_group('model')
    add_general_argument(evaluate_general_parser)
    add_evaluation_io_argument(evaluate_io_parser)
    add_feature_argument(evaluate_feature_parser)
    add_model_argument(evaluate_model_parser)

    args = parser.parse_args()
    if args.command == 'train':
        validate_general_argument(args, parser)
        validate_training_io_argument(args, parser)
        validate_feature_argument(args, parser)
        validate_model_argument(args, parser)
        validate_training_argument(args, parser)
    elif args.command == 'predict':
        validate_general_argument(args, parser)
        validate_prediction_io_argument(args, parser)
        validate_feature_argument(args, parser)
        validate_model_argument(args, parser)
    elif args.command == 'evaluate':
        validate_general_argument(args, parser)
        validate_evaluation_io_argument(args, parser)
        validate_feature_argument(args, parser)
        validate_model_argument(args, parser)
    return args

def train(args):
    # build dataset
    train_dataset = FolderTuple(args.train_dir, args.sr, args.segment_duration)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    if args.validation_dir is not None:
        validation_dataset = FolderTuple(
            args.validation_dir, args.sr, args.segment_duration)
        validation_loader = torch.utils.data.DataLoader(
            validation_dataset, batch_size=args.batch_size, shuffle=False
        )

    # build (and load) a model
    model = ChimeraMagPhasebook(
        args.bin_num,
        len(args.train_dir),
        args.embedding_dim,
        N=args.n_hidden,
        residual_base=args.residual
    )
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    initial_epoch = 0
    if args.input_checkpoint is not None:
        checkpoint = torch.load(args.input_checkpoint)
        initial_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(args.device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    elif args.input_model is not None:
        model.load_state_dict(torch.load(args.input_model))
        model.to(args.device)
    else:
        model.to(args.device)

    # train and validation loop
    epoch = initial_epoch
    for epoch in range(initial_epoch+1, initial_epoch+args.epoch+1):
        sum_loss = 0
        total_batch = 0
        ave_loss = 0
        last_output_len = 0
        if not args.sync:
            train_dataset.shuffle()
        model.train()
        for step, batch in enumerate(train_loader, 1):
            batch = batch.to(args.device)
            x_in = make_x_in(batch, args)
            y_pred = forward(model, x_in, args)
            loss = compute_loss(x_in, y_pred, args)
            sum_loss += loss.item()
            total_batch += batch.shape[0]
            ave_loss = sum_loss / total_batch
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Print learning statistics
            curr_output = f'\repoch {epoch} step {step} loss={ave_loss}'
            sys.stdout.write('\r' + ' ' * last_output_len)
            sys.stdout.write(curr_output)
            sys.stdout.flush()
            last_output_len = len(curr_output)

        if args.validation_dir is not None:
            if not args.sync:
                validation_dataset.shuffle()
            model.eval()
            with torch.no_grad():
                sum_val_loss = 0
                total_batch = 0
                for batch in validation_loader:
                    batch = batch.to(args.device)
                    x_in = make_x_in(batch, args)
                    y_pred = forward(model, x_in, args)
                    loss = compute_loss(x_in, y_pred, args)
                    sum_val_loss += loss.item()
                    total_batch += batch.shape[0]
            ave_val_loss = sum_val_loss / total_batch
            sys.stdout.write('\r' + ' ' * last_output_len)
            sys.stdout.write(f'\repoch {epoch} loss={ave_loss} val={ave_val_loss}\n')
    # end of epoch loop

    # save
    model.eval()
    if args.output_model is not None:
        torch.save(model.state_dict(), args.output_model)
    if args.output_checkpoint is not None:
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            },
            args.output_checkpoint
        )

def predict(args):
    # load audio information
    batch, orig_sr = torchaudio.load(args.input_file)
    batch = torch.Tensor(
        resampy.resample(batch.numpy(), orig_sr, args.sr, axis=-1)
    )
    if batch.dim() == 1:
        batch = batch.unsqueeze(0)

    # build (and load) a model
    model = ChimeraMagPhasebook(
        args.bin_num,
        len(args.output_files),
        args.embedding_dim,
        N=args.n_hidden,
        residual_base=args.residual
    )
    if args.input_checkpoint is not None:
        checkpoint = torch.load(args.input_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
    elif args.input_model is not None:
        model.load_state_dict(torch.load(args.input_model))

    # predict and save
    shat = predict_waveform(model, batch, args)
    for f, s in zip(args.output_files, shat):
        torchaudio.save(f, s, sample_rate=args.sr)

def evaluate(args):
    # build dataset
    dataset = FolderTuple(args.data_dir, args.sr, args.segment_duration)
    if not args.sync:
        dataset.shuffle()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False
    )

    # load a model
    model = ChimeraMagPhasebook(
        args.bin_num,
        len(args.data_dir),
        args.embedding_dim,
        N=args.n_hidden,
        residual_base=args.residual
    )
    if args.input_checkpoint is not None:
        checkpoint = torch.load(args.input_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
    elif args.input_model is not None:
        model.load_state_dict(torch.load(args.input_model))
    model.to(args.device)
    model.eval()

    if args.permutation_free:
        eval_snr = torchchimera.metrics.permutation_free(
            torchchimera.metrics.eval_snr, aggregate_functionn=max
        )
        eval_si_sdr = torchchimera.metrics.permutation_free(
            torchchimera.metrics.eval_si_sdr, aggregate_function=max
        )
    else:
        eval_snr = torchchimera.metrics.eval_snr
        eval_si_sdr = torchchimera.metrics.eval_si_sdr

    # evaluation loop
    if args.output_file is None:
        of = sys.stdout
    else:
        of = open(args.output_file, 'w')
    print('data,channel,snr,si-sdr', file=of)
    with torch.no_grad():
        for data_i, s in enumerate(map(lambda s: s.unsqueeze(0), dataset), 1):
            s = s.to(args.device)
            x = s.sum(dim=1)
            shat = predict_waveform(model, x, args)
            waveform_length = min(s.shape[-1], shat.shape[-1])
            s = s[:, :, :waveform_length]
            shat = shat[:, :, :waveform_length]

            snr = eval_snr(shat, s)[0]
            si_sdr = eval_si_sdr(shat, s)[0]
            for channel_i, (_snr, _si_sdr) in enumerate(zip(snr, si_sdr), 1):
                print(f'{data_i},{channel_i},{_snr},{_si_sdr}', file=of)
    of.close()

def main():
    args = parse_args()
    if args.command == 'train':
        train(args)
    elif args.command == 'predict':
        predict(args)
    elif args.command == 'evaluate':
        evaluate(args)

if __name__ == '__main__':
    main()
