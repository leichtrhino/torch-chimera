#!/usr/bin/env python

import torch
from losses import *

batch_size = 16
time = 300
freq_bin = 129
embd_dim = 20
n_channels = 2
waveform_length = 8000

embd = torch.randn(batch_size, time*freq_bin, embd_dim)
label = torch.randn(batch_size, time*freq_bin, n_channels)

mask = torch.randn(batch_size, n_channels, freq_bin, time)
mixture = torch.randn(batch_size, freq_bin, time, 2)
sources = torch.randn(batch_size, n_channels, freq_bin, time, 2)

waveform_true = torch.randn(batch_size, n_channels, waveform_length)
waveform_pred = torch.randn(batch_size, n_channels, waveform_length)

print(loss_dc(embd, label))
print(loss_dc_whitend(embd, label))

print(loss_mi_msa(mask, mixture, sources))
print(loss_mi_tpsa(mask, mixture, sources))

print(loss_wa(waveform_pred, waveform_true))
