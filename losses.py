import torch
from itertools import permutations

# loss functions for deep clustering head
# embd: (batch_size, time*freq_bin, embd_dim)
# label: (batch_size, time*freq_bin, n_channels)
def loss_dc(embd, label):
    return torch.sum(torch.matmul(embd.transpose(1, 2), embd) ** 2) \
        + torch.sum(torch.matmul(label.transpose(1, 2), label) ** 2) \
        - 2 * torch.sum(torch.matmul(embd.transpose(1, 2), label) ** 2)

def loss_dc_whitend(embd, label):
    C = label.shape[2]
    D = embd.shape[2]
    VtV = torch.matmul(embd.transpose(1, 2), embd) + 1e-9 * torch.eye(D)
    VtY = torch.matmul(embd.transpose(1, 2), label)
    YtY = torch.matmul(label.transpose(1, 2), label) + 1e-9 * torch.eye(C)
    return embd.shape[0] * D - torch.trace(torch.sum(
        torch.matmul(torch.matmul(torch.matmul(
            VtV.inverse(), VtY), YtY.inverse()), VtY.transpose(1, 2)),
        0
    ))

# loss functions for mask inference head
# mask: (batch_size, n_channels, freq_bin, time)
# source: (batch_size, n_channels, freq_bin, time, 2)
# mixture: (batch_size, freq_bin, time, 2)
# source and mixture are obtained from torch.stft
def loss_mi_msa(mask, mixture, sources):
    C = mask.shape[1]
    abs_comp = lambda X: torch.sqrt(torch.sum(X**2, -1))
    phase_comp = lambda X: torch.atan2(*X.split(1, dim=-1)[::-1]).squeeze()
    abs_X = abs_comp(mixture)
    abs_S = abs_comp(sources)
    return sum(min(
        torch.sum((torch.stack(M) * _abs_X - _abs_S) ** 2)
        for M in permutations(_mask)
    ) for _mask, _abs_X, _abs_S in zip(mask, abs_X, abs_S))

def loss_mi_tpsa(mask, mixture, sources, gamma=1, L=1):
    C = mask.shape[1]
    abs_comp = lambda X: torch.sqrt(torch.sum(X**2, -1))
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
        return sum(min(
            torch.sum(torch.abs(torch.stack(M) * _abs_X - _spectrum))
            for M in permutations(_mask)
        ) for _mask, _abs_X, _spectrum in zip(mask, abs_X, spectrum))
    elif L == 2:
        return sum(min(
            torch.sum((torch.stack(M) * _abs_X - _spectrum) ** L)
            for M in permutations(_mask)
        ) for _mask, _abs_X, _spectrum in zip(mask, abs_X, spectrum))
    else:
        raise NotImplementedError()

# loss for waveform approximation
# source_pred: (batch_size, n_channels, waveform_length)
# source_true: (batch_size, n_channels, waveform_length)
def loss_wa(source_pred, source_true):
    return sum(min(
            torch.sum(torch.abs(torch.stack(s) - _source_true))
            for s in permutations(_source_pred)
        ) for _source_pred, _source_true in zip(source_pred, source_true))
