
import os
import sys

import torch
import torchaudio
import resampy

try:
    import torchchimera
except:
    # attempts to import local module
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    import torchchimera
from torchchimera.models.chimera import ChimeraMagPhasebook

from _training_common import predict_waveform

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
    shat = predict_waveform(model, batch, args.stft_setting)
    if shat.dim() == 3:
        shat = shat.squeeze(0)
    for f, s in zip(args.output_files, shat):
        torchaudio.save(f, s, sample_rate=args.sr)

