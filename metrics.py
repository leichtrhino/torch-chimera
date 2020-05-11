
import torch
from itertools import permutations

# s_pred: (batch_size, n_channel, waveform_length)
# s_true: (batch_size, n_channel, waveform_length)
def eval_snr(s_pred, s_true):
    return torch.stack([max(
        (10 * torch.log10(torch.clamp(
            torch.sum(_s_true**2, dim=-1) /
            torch.sum((_s_true-torch.stack(s))**2, dim=-1).clamp(min=1e-24),
            min=1e-24
        )) for s in permutations(_s_pred)),
        key=lambda t: torch.sum(t)
    ) for _s_pred, _s_true in zip(s_pred, s_true)])

# s_pred: (batch_size, n_channel, waveform_length)
# s_true: (batch_size, n_channel, waveform_length)
def eval_si_sdr(s_pred, s_true):
    scale_s = lambda shat, s: torch.unsqueeze(
        torch.sum(torch.stack(shat) * s, dim=-1) /
        torch.sum(s**2, dim=-1).clamp(1e-24),
        dim=-1
    ) * s
    return torch.stack([max(
        (10 * torch.log10(torch.clamp(
            torch.sum(scale_s(s, _s_true)**2, dim=-1) /
            torch.sum(
                (scale_s(s, _s_true) - torch.stack(s))**2, dim=-1
            ).clamp(min=1e-24),
            min=1e-24
        )) for s in permutations(_s_pred)),
        key=lambda t: torch.sum(t)
    ) for _s_pred, _s_true in zip(s_pred, s_true)])

