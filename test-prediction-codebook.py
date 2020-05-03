#!/usr/bin/env python

import sys
import math
import torch
import torchaudio
from datasets import DSD100, MixTransform
from layers import MisiLayer
from models import ChimeraMagPhasebook

def main():
    model_file = 'model_misi.pth'
    batch_size = 24
    batch_idx = 2
    orig_freq = 44100
    target_freq = 16000
    seconds = 5

    n_fft = 512
    win_length = 512
    hop_length = 128

    freq_bins, spec_time, _ = torch.stft(
        torch.Tensor(seconds * target_freq), n_fft, hop_length, win_length
    ).shape

    stft = lambda x: torch.stft(
        x.reshape(x.shape[:-1].numel(), seconds * target_freq),
        n_fft, hop_length, win_length
    ).reshape(*x.shape[:-1], freq_bins, spec_time, 2)
    istft = lambda X: torchaudio.functional.istft(
        X.reshape(X.shape[:-3].numel(), freq_bins, spec_time, 2),
        self.n_fft, self.hop_length, self.win_length
    ).reshape(*X.shape[:-3], waveform_length)
    comp_mul = lambda X, Y: torch.stack(
        (X.unbind(-1)[0] * Y.unbind(-1)[0] - X.unbind(-1)[1] * Y.unbind(-1)[1],
         X.unbind(-1)[0] * Y.unbind(-1)[1] + X.unbind(-1)[1] * Y.unbind(-1)[0]),
        dim=-1
    )

    dataset = DSD100(
        '/Volumes/Buffalo 2TB/Datasets/DSD100', 'Test', seconds * orig_freq)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=8, shuffle=False)
    transforms = [
        MixTransform([(0, 1, 2), 3, (0, 1, 2, 3)]),
        lambda x: x.reshape(x.shape[0] * 3, seconds * orig_freq),
        torchaudio.transforms.Resample(orig_freq, target_freq),
        lambda x: torch.stft(x, n_fft, hop_length, win_length),
        lambda x: x.reshape(x.shape[0] // 3, 3, freq_bins, spec_time, 2),
    ]
    def transform(x):
        for t in transforms:
            x = t(x)
        return x

    model = ChimeraMagPhasebook(freq_bins, spec_time, 2, 20, N=600)
    model.load_state_dict(torch.load(model_file))
    misilayer = MisiLayer(n_fft, hop_length, win_length, 5)

    #batch = transform(next(iter(dataloader)))
    batch = transform(dataset[list(range(
        batch_idx*batch_size, (batch_idx+1)*batch_size
    ))])
    S = batch[:, :2, :, :, :]
    X = batch[:, 2, :, :, :]
    X_abs = torch.sqrt(torch.sum(X**2, dim=-1))
    X_phase = X / X_abs.clamp(min=1e-9).unsqueeze(-1)
    x = torchaudio.functional.istft(
        X, n_fft, hop_length, win_length
    )

    _, (com,) = model(torch.log10(X_abs.clamp(min=1e-9)), outputs=['com'])
    com = com.detach()
    Shat = comp_mul(com, X.unsqueeze(1))
    _, shat = misiLayer(Shat, x)
    shat = shat.transpose(0, 1).reshape(2, batch_size * seconds * target_freq)

    s = torchaudio.functional.istft(
        S.reshape(batch_size*2, freq_bins, spec_time, 2),
        n_fft, hop_length, win_length
    ).reshape(batch_size, 2, seconds * target_freq).transpose(0, 1) \
    .reshape(2, batch_size * seconds * target_freq)

    for i_channel, (_s, _shat) in enumerate(zip(s, shat)):
        torchaudio.save(f's_{i_channel}.wav', _s, target_freq)
        torchaudio.save(f'shat_{i_channel}.wav', _shat, target_freq)

if __name__ == '__main__':
    main()
