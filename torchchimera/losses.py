import torch
from math import pi
from itertools import permutations

# loss functions for deep clustering head
# embd: (batch_size, time*freq_bin, embd_dim)
# label: (batch_size, time*freq_bin, n_channels)
# weight: (batch_size, time*freq_bin)
def loss_dc(embd, label, weight=None):
    if type(weight) == torch.Tensor:
        weight = torch.sqrt(weight).unsqueeze(-1)
        embd = embd * weight
        label = label * weight
    return torch.sum(embd.transpose(1, 2).bmm(embd) ** 2) \
        + torch.sum(label.transpose(1, 2).bmm(label) ** 2) \
        - 2 * torch.sum(embd.transpose(1, 2).bmm(label) ** 2)

def loss_dc_deep_lda(embd, label, weight=None):
    if type(weight) == torch.Tensor:
        weight = torch.sqrt(weight).unsqueeze(-1)
        embd = embd * weight
        label = label * weight
    C = label.shape[2]
    YtV = label.transpose(1, 2).bmm(embd)
    YtY = label.transpose(1, 2).bmm(label) + 1e-24 * torch.eye(C)
    return torch.sum((embd - label.bmm(YtY.inverse().bmm(YtV))) ** 2)

def loss_dc_whitened(embd, label, weight=None):
    if type(weight) == torch.Tensor:
        weight = torch.sqrt(weight).unsqueeze(-1)
        embd = embd * weight
        label = label * weight
    C = label.shape[2]
    D = embd.shape[2]
    VtV = embd.transpose(1, 2).bmm(embd) + 1e-24 * torch.eye(D)
    VtY = embd.transpose(1, 2).bmm(label)
    YtY = label.transpose(1, 2).bmm(label) + 1e-24 * torch.eye(C)
    return embd.shape[0] * D - torch.trace(torch.sum(
        VtV.inverse().bmm(VtY).bmm(YtY.inverse()).bmm(VtY.transpose(1, 2)),
        dim=0
    ))

def permutation_free(loss_function):
    def _loss_function(*args, **kwargs):
        return sum(
            min(
                loss_function(
                    *[a.unsqueeze(0) for a in (torch.stack(p), *_args[1:])],
                    **kwargs
                ) for p in permutations(_args[0])
            ) for _args in zip(*args)
        )
    return _loss_function

# loss functions for mask inference head
# mask: (batch_size, n_channels, freq_bin, time)
# source: (batch_size, n_channels, freq_bin, time, 2)
# mixture: (batch_size, freq_bin, time, 2)
# source and mixture are obtained from torch.stft
def loss_mi_msa(mask, mixture, sources):
    C = mask.shape[1]
    abs_comp = lambda X: torch.sqrt(torch.sum(X**2, -1).clamp(min=1e-12))
    phase_comp = lambda X: torch.atan2(*X.split(1, dim=-1)[::-1]).squeeze()
    abs_X = abs_comp(mixture)
    abs_S = abs_comp(sources)
    return torch.sum((mask * abs_X - abs_S) ** 2)

def loss_mi_tpsa(mask, mixture, sources, gamma=1, L=1):
    C = mask.shape[1]
    abs_comp = lambda X: torch.sqrt(torch.sum(X**2, -1).clamp(min=1e-12))
    phase_comp = lambda X: torch.atan2(*X.split(1, dim=-1)[::-1]).squeeze(-1)
    abs_X = abs_comp(mixture.unsqueeze(1))
    phase_X = phase_comp(mixture.unsqueeze(1))
    abs_S = abs_comp(sources)
    phase_S = phase_comp(sources)
    spectrum = torch.min(
        input=torch.max(
            input=abs_S * torch.cos(phase_S - phase_X),
            other=torch.zeros_like(abs_S)
        ),
        other=gamma*abs_X
    )

    if L == 1:
        return torch.sum(torch.abs(mask * abs_X - spectrum))
    elif L == 2:
        return torch.sum((mask * abs_X - spectrum) ** L)
    else:
        raise NotImplementedError()

# loss for waveform approximation
# source_pred: (batch_size, n_channels, waveform_length)
# source_true: (batch_size, n_channels, waveform_length)
def loss_wa(source_pred, source_true):
    return torch.sum(torch.abs(source_pred - source_true))

# loss function for spectrogram (complex domain)
# com_pred: (batch_size, n_channels, freq_bin, spec_time, 2)
# mixture: (batch_size, freq_bin, spec_time, 2)
# sources: (batch_size, n_channels, freq_bin, spec_time, 2)
def loss_csa(com_pred, mixture, sources, L=1):
    comp_mul = lambda X, Y: torch.stack(
        (X.unbind(-1)[0] * Y.unbind(-1)[0] - X.unbind(-1)[1] * Y.unbind(-1)[1],
         X.unbind(-1)[0] * Y.unbind(-1)[1] + X.unbind(-1)[1] * Y.unbind(-1)[0]),
        dim=-1
    )
    abs_comp = lambda X: torch.sqrt(torch.sum(X**2, -1).clamp(min=1e-12))
    phase_comp = lambda X: torch.atan2(*X.split(1, dim=-1)[::-1]).squeeze(-1)
    source_pred = comp_mul(com_pred, mixture.unsqueeze(1))

    if L == 1:
        return torch.sum(abs_comp(source_pred - sources))
    elif L == 2:
        return torch.sum(abs_comp(source_pred - sources) ** 2)
    else:
        raise NotImplementedError()
