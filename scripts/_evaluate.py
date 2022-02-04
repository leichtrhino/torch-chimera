
import os
import sys
import math
import numpy as np
import museval

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
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
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
    if args.batch_size <= 0:
        parser.error('batch size must be positive')
    return args

def evaluate(args):
    # build dataset
    dataset = FolderTuple(args.data_dir, args.sr, args.segment_duration)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False
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

    # evaluation loop
    if args.output_file is None:
        of = sys.stdout
    else:
        of = open(args.output_file, 'w')

    # evaluation
    sample_segment_length = int(args.sr * args.segment_duration)
    print('sample_i,sdr,isr,sir,sar', file=args.output_file)
    for sample_i, sample in enumerate(loader):
        sample = sample.flatten(0, -2)
        sample_sum = sample.sum(dim=0)

        # divide into batches
        orig_length = sample_sum.shape[-1]
        sample_num = math.ceil(sample_sum.shape[-1] / sample_segment_length)

        batch = torch.cat((
            sample_sum,
            torch.zeros(
                *sample_sum.shape[:-1],
                sample_segment_length * sample_num - sample_sum.shape[-1]
            )
        ), dim=-1).reshape(
            *sample_sum.shape[:-1], sample_num, sample_segment_length
        )

        out_tensor = None
        for batch_i in range(0, batch.shape[0], args.batch_size):
            batch_end_i = min(batch_i + args.batch_size, batch.shape[0])
            b = batch[batch_i:batch_end_i]
            with torch.no_grad():
                _, _, s_hat, _ = model(b.to(args.device))
                s_hat = s_hat.cpu().transpose(0, 1).flatten(1, -1)

            if out_tensor is None:
                out_tensor = s_hat
            else:
                out_tensor = torch.cat((out_tensor, s_hat), dim=-1)

        # align input and output
        print(sample.shape, out_tensor.shape)
        out_length = min(orig_length, out_tensor.shape[-1])
        sample = sample[..., :out_length]
        s_hat = out_tensor[..., :out_length]
        # find the combination of signs that satisfies minimum error
        s_hat_sig = []
        for sample_, s_hat_ in zip(sample, s_hat):
            if torch.sum((sample_ - s_hat_) ** 2) \
               < torch.sum((sample_ + s_hat_) ** 2):
                s_hat_sig.append(s_hat_)
            else:
                s_hat_sig.append(-s_hat_)
        s_hat = torch.stack(s_hat_sig, dim=0)

        # evaluate by museval.evaluate
        SDR, ISR, SIR, SAR = museval.evaluate(
            sample.unsqueeze(-1).numpy(),
            s_hat.unsqueeze(-1).numpy(),
            win=args.sr * 1,
            hop=args.sr * 1
        )

        # write to output
        for sdr, isr, sir, sar in zip(SDR, ISR, SIR, SAR):
            print(
                f'{sample_i},{np.nanmedian(sdr)},{np.nanmedian(isr)},{np.nanmedian(sir)},{np.nanmedian(sar)}',
                file=of
            )

    of.close()

