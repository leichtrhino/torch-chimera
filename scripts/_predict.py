
import os
import sys
import math

import torch
import torchaudio
import resampy

try:
    import torchchimera
except:
    # attempts to import local module
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    import torchchimera

from _model_io import load_model
from _training_common import AdaptedChimeraMagPhasebook

def add_prediction_io_argument(parser):
    parser.add_argument('--input-file', required=True, help='input wav file')
    parser.add_argument('--output-files', required=True, nargs='+', help='output wav files')
    parser.add_argument('--input-checkpoint', help='input checkpoint file')
    parser.add_argument('--log-file', help='log file')
    parser.add_argument('--batch-size', type=int, default=32)
    return parser

def validate_prediction_io_argument(args, parser):
    if not os.path.isfile(args.input_file):
        parser.error(f'"{args.input_file}" is not a file')
    if args.input_checkpoint and not os.path.isfile(args.input_checkpoint):
        parser.error(f'input checkpoint "{args.input_checkpoint}" is not a file')
    return args

def predict(args):
    # load audio information
    batch, orig_sr = torchaudio.load(args.input_file)
    batch = torch.Tensor(
        resampy.resample(batch.numpy(), orig_sr, args.sr, axis=-1)
    )
    if batch.dim() == 1:
        batch = batch.unsqueeze(0)
    batch = batch.to(args.device)
    if args.segment_duration is not None:
        orig_len = batch.shape[-1]
        segment_length = int(args.segment_duration * args.sr)
        sample_size = math.ceil(batch.shape[-1] / segment_length)
        batch = torch.cat((
            batch,
            torch.zeros(
                *batch.shape[:-1],
                sample_size * segment_length - orig_len,
                device=args.device)
        ), dim=-1)
        batch = batch.reshape(
            batch.shape[0],
            sample_size,
            segment_length)

    # load a model
    model, update_args = load_model(
        args.input_checkpoint, None,
        stft_setting=args.stft_setting
    )
    if args.bin_num != update_args['bin_num']:
        bin_num = checkpoint['model']['parameter']['bin_num']
        raise RuntimeError(
            'the number of fft bin of input model and parameter are different '
            f'--n-fft {(bin_num-1)*2} would work'
        )
    if len(args.output_files) != update_args['n_channel']:
        raise RuntimeError(
            'the number of channels of the input model '
            'and the output files are different'
        )
    model.to(args.device)
    model.eval()

    # predict and save
    if args.segment_duration is not None:
        shats = []
        for batch_channel in batch:
            shat = None
            for batch_i in range(0, sample_size, args.batch_size):
                batch_end = min(batch_i + args.batch_size, sample_size)
                minibatch = batch_channel[batch_i:batch_end]
                _, _, _shat, _ = model(minibatch)
                if shat is None:
                    shat = _shat
                else:
                    shat = torch.cat((shat, _shat), dim=0)
            shats.append(
                shat.transpose(0, 1).reshape(
                    shat.shape[1], sample_size * segment_length))
        shat = torch.stack(shats, dim=0)[:orig_len]
    else:
        _, _, shat, _ = model(batch)

    if shat.shape[0] > 1:
        shat = shat.transpose(0, 1)
    else:
        shat = shat.squeeze(0)
    shat = shat.to('cpu')
    for f, s in zip(args.output_files, shat):
        torchaudio.save(f, s, sample_rate=args.sr)

