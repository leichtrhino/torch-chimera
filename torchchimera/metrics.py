
import torch
from itertools import permutations

# s_pred: (batch_size, n_channel, waveform_length)
# s_true: (batch_size, n_channel, waveform_length)
def eval_snr(s_pred, s_true):
    return 10 * torch.log10(
        torch.sum(s_true**2, dim=-1) /
        torch.sum((s_true - s_pred)**2, dim=-1)
    )

# s_pred: (batch_size, n_channel, waveform_length)
# s_true: (batch_size, n_channel, waveform_length)
def eval_si_sdr(s_pred, s_true):
    scale_s = torch.sum(s_pred * s_true, dim=-1, keepdims=True) /\
        torch.sum(s_true**2, dim=-1, keepdims=True) * s_true
    return 10 * torch.log10(
        torch.sum(scale_s ** 2, axis=-1) /
        torch.sum((scale_s - s_pred) ** 2, axis=-1)
    )

def permutation_free(eval_function):
    def _eval_function(*args, **kwargs):
        return [
            max(
                (p for p in permutations(_args[0])),
                key=lambda p: sum(eval_function(*[a.unsqueeze(0) for a in (torch.stack(p), *_args[1:])], **kwargs))
            )
            for _args in zip(*args)
        ]
    return _eval_function

