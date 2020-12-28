
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
from torchchimera.models.chimera import ChimeraMagPhasebook
from torchchimera.metrics import eval_snr
from torchchimera.metrics import eval_si_sdr

from _training_common import predict_waveform
from _training_common import Stft
from _training_common import Istft

def add_evaluation_io_argument(parser):
    parser.add_argument('--data-dir', nargs='+', required=True, help="directory of validation dataset")
    parser.add_argument('--cutoff-rms', type=float, help='cutoff RMS (in db)')
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
            if args.cutoff_rms:
                window = torch.sqrt(torch.hann_window(args.n_fft)).to(args.device)
                stft = Stft(args.n_fft, args.hop_length, args.win_length, window)
                S = stft(s.squeeze(0))
                S_pow = torch.sum(stft(s.squeeze(0))**2, dim=-1)
                rms = 10 * torch.log10(torch.mean(S_pow, dim=1))
                is_no_silence = torch.all(rms > args.cutoff_rms, dim=0)
                S = S[:, :, is_no_silence, :]
                if S.shape[2] < 4: # 4: n_fft // hop_length
                    continue
                istft = Istft(args.n_fft, args.hop_length, args.win_length, window)
                s = istft(S).unsqueeze(0)

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

