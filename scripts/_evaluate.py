
import os
import sys

import torch
try:
    import torchchimera
except:
    # attempts to import local module
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    import torchchimera
from torchchimera.datasets import FolderTuple
from torchchimera.metrics import eval_snr
from torchchimera.metrics import eval_si_sdr

from _model_io import load_model
from _training_common import AdaptedChimeraMagPhasebook
from _training_common import exclude_silence

def add_evaluation_io_argument(parser):
    parser.add_argument('--data-dir', nargs='+', required=True, help="directory of validation dataset")
    parser.add_argument('--input-checkpoint', help='input checkpoint file')
    parser.add_argument('--output-file', help='output file')
    parser.add_argument('--log-file', help='log file')
    parser.add_argument('--permutation-free', action='store_true', help='enable permutation-free evaluation function')
    return parser

def validate_evaluation_io_argument(args, parser):
    for d in args.data_dir:
        if not os.path.isdir(d):
            parser.error(f'"{d}" is not a directory')
    if args.input_checkpoint and not os.path.isfile(args.input_checkpoint):
        parser.error(f'input checkpoint "{args.input_checkpoint}" is not a file')
    return args

def evaluate(args):
    # build dataset
    dataset = FolderTuple(args.data_dir, args.sr, args.segment_duration)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False
    )

    # load a model
    model, update_args = load_model(
        args.input_checkpoint, 'ChimeraMagPhasebook',
        stft_setting=args.stft_setting
    )
    if args.bin_num != update_args['bin_num']:
        bin_num = checkpoint['model']['parameter']['bin_num']
        raise RuntimeError(
            'the number of fft bin of input model and parameter are different '
            f'--n-fft {(bin_num-1)*2} would work'
        )
    if len(args.data_dir) != update_args['n_channel']:
        raise RuntimeError(
            'the number of channels of the input model '
            'and the output files are different'
        )
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
    print('segment,channel,snr,si-sdr', file=of)
    with torch.no_grad():
        for data_i, s in enumerate(map(lambda s: s.unsqueeze(0), dataset), 1):
            scale = torch.sqrt(
                s.shape[-1] / torch.sum(s**2, dim=-1).clamp(min=1e-32)
            )
            scale_mix = 1. / torch.max(
                torch.sum(scale.unsqueeze(-1) * s, dim=1).abs(), dim=-1
            )[0]
            scale_mix = torch.min(scale_mix, torch.ones_like(scale_mix))
            scale *= scale_mix.unsqueeze(-1)
            s *= scale.unsqueeze(-1) * 0.98
            s = s.to(args.device)

            _, _, shat, _ = model(s.sum(dim=1))
            waveform_length = min(s.shape[-1], shat.shape[-1])
            s = s[:, :, :waveform_length]
            shat = shat[:, :, :waveform_length]

            snr = eval_snr(shat, s)[0]
            si_sdr = eval_si_sdr(shat, s)[0]
            for channel_i, (_snr, _si_sdr) in enumerate(zip(snr, si_sdr), 1):
                print(f'{data_i},{channel_i},{_snr},{_si_sdr}', file=of)
    of.close()

