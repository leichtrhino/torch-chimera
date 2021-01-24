
import os
import sys
from itertools import chain

import torch

try:
    import torchchimera
except:
    # attempts to import local module
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    import torchchimera
from torchchimera.datasets import FolderTuple

from _model_io import build_model
from _model_io import build_optimizer
from _model_io import load_model
from _model_io import load_optimizer
from _model_io import save_checkpoint
from _training_common import AdaptedChimeraMagPhasebook
from _training_common import compute_loss

def add_training_io_argument(parser):
    parser.add_argument('--train-dir', nargs='+', required=True, help='directory of training dataset')
    parser.add_argument('--validation-dir', nargs='*', help="directory of validation dataset")
    parser.add_argument('--output-model', help='output model file')
    parser.add_argument('--output-checkpoint', help='output checkout file')
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
    if args.model_type is None and args.input_checkpoint is None:
        parser.error('--model-type is required when --input-checkpoint is not provided')
    if args.input_checkpoint and not os.path.isfile(args.input_checkpoint):
        parser.error(f'input checkpoint "{args.input_checkpoint}" is not a file')
    if not args.output_checkpoint:
        parser.error('--output-checkpoint must be provided')
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
    if args.input_checkpoint is None:
        args.n_channel = len(args.train_dir)
        model = build_model(
            args.model_type,
            bin_num=args.bin_num,
            n_channel=args.n_channel,
            embedding_dim=args.embedding_dim,
            n_hidden=args.n_hidden,
            residual=args.residual,
            n_lstm_layer=args.n_lstm_layer,
            n_misi_layer=args.n_misi_layer,
            stft_setting=args.stft_setting,
        )
        model.to(args.device)
        optimizer = build_optimizer(model, lr=args.lr)
        initial_epoch = 0
    else:
        # load model from file
        model, update_args = load_model(
            args.input_checkpoint,
            args.model_type,
            n_misi_layer=args.n_misi_layer,
            stft_setting=args.stft_setting
        )
        if args.bin_num != update_args['bin_num']:
            bin_num = update_args['bin_num']
            raise RuntimeError(
                'the number of fft bin of input model and parameter are different '
                f'--n-fft {(bin_num-1)*2} would work'
            )
        if len(args.train_dir) != update_args['n_channel']:
            raise RuntimeError(
                'the number of channels of the input model '
                'and the dataset are different'
            )
        for k, v in update_args.items():
            vars(args)[k] = v
        model.to(args.device)

        # load optimizer from file
        try:
            optimizer, initial_epoch, update_args = load_optimizer(
                args.input_checkpoint, model)
            if args.loss_function is None or\
               args.loss_function == update_args['loss_function']:
                for k, v in update_args.items():
                    vars(args)[k] = v
            else:
                raise RuntimeError()
        except:
            # build optimizer from scratch
            optimizer = build_optimizer(model, lr=args.lr)
            initial_epoch = 0

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
            if not args.sync:
                scale = 10 ** (torch.rand(*batch.shape[:-1]) * -10 / 10)
                scale = torch.sqrt(
                    batch.shape[-1] * scale**2 /
                    torch.sum(batch**2, dim=-1).clamp(min=1e-32)
                )
                scale_mix = 1. / torch.max(
                    torch.sum(scale.unsqueeze(-1) * batch, dim=1).abs(), dim=-1
                )[0]
                scale_mix = torch.min(scale_mix, torch.ones_like(scale_mix))
                scale *= scale_mix.unsqueeze(-1)
                batch *= scale.unsqueeze(-1) * 0.98

            batch = batch.to(args.device)
            y_pred = model(batch.sum(dim=1))
            loss = compute_loss(batch, y_pred, args.stft_setting,
                                args.loss_function, args.permutation_free)
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
                    y_pred = model(batch.sum(dim=1))
                    loss = compute_loss(batch, y_pred, args.stft_setting,
                                        args.loss_function, args.permutation_free)
                    sum_val_loss += loss.item()
                    total_batch += batch.shape[0]
            ave_val_loss = sum_val_loss / total_batch
            sys.stdout.write('\r' + ' ' * last_output_len)
            sys.stdout.write(f'\repoch {epoch} loss={ave_loss} val={ave_val_loss}\n')
    # end of epoch loop

    # save
    model.eval()
    if args.output_checkpoint is not None:
        del vars(args)['epoch']
        save_checkpoint(
            model, optimizer, args.output_checkpoint, epoch=epoch, **vars(args))
