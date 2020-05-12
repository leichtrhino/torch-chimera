#!/usr/bin/env python
import torch
import torchaudio
import matplotlib.pyplot as plt
from math import pi
from layers import MisiNetwork, TrainableMisiNetwork

def main():
    window_length, n_fft = 16000, 512
    s = torch.stack([
        torch.sin(2 * pi * (i/2)*220 * torch.linspace(0, 1, window_length))
        for i in range(1, 11)
    ]).reshape(5, 2, window_length)
    x = torch.sum(s, dim=1)
    #model = MisiNetwork(n_fft, n_fft//4, n_fft, 5)
    model = TrainableMisiNetwork(n_fft, 0)
    S = torch.stft(
        s.reshape(10, window_length),
        n_fft, window=torch.hann_window(n_fft)
    )
    _, freq_bins, spec_time, _ = S.shape
    S = S.reshape(5, 2, freq_bins, spec_time, 2)
    X = torch.stft(x, n_fft, window=torch.hann_window(n_fft))
    Smag, Xmag = S.norm(2, -1), X.norm(2, -1)
    mask = 10 ** (torch.log10(Smag.clamp(min=1e-36)) -\
        torch.log10(Xmag.clamp(min=1e-24).unsqueeze(1)))
    sbar = torchaudio.functional.istft(
        (mask.unsqueeze(-1) * X.unsqueeze(1))
        .reshape(10, freq_bins, spec_time, 2),
        n_fft,
        window=torch.hann_window(n_fft)
    ).reshape(5, 2, window_length)
    shat = model(mask, x).detach()

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
    ax1.plot(x[2, :1000])
    ax2.plot(s[2, 0, :1000])
    ax3.plot(sbar[2, 0, :1000])
    ax4.plot(shat[2, 0, :1000])
    plt.show()

if __name__ =='__main__':
    main()
