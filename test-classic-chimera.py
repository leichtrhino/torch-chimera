#!/usr/bin/env python

import torch
from models import ClassicChimera

batch_size = 16
time = 300
freq_bin = 129
embd_dim = 20
n_channels = 2
waveform_length = 8000

model = ClassicChimera(freq_bin, time, n_channels, embd_dim)
x = torch.randn(batch_size, freq_bin, time)
embd_pred, mask_pred = model(x)

print(embd_pred.shape, mask_pred.shape)
assert (embd_pred.shape == (batch_size, time*freq_bin, embd_dim)), \
    'shape of embedding mismatch'
assert (mask_pred.shape == (batch_size, n_channels, freq_bin, time)), \
    'shape of mask mismatch'

