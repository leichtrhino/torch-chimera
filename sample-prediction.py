#!/usr/bin/env python
import sys
import torch
import torchaudio
import numpy as np
import sklearn.cluster
import matplotlib.pyplot as plt
from torchchimera.datasets import DSD100
from torchchimera.transforms import MixTransform
from torchchimera.transforms import Resample
from torchchimera.transforms import Compose
from torchchimera.submodels import MisiNetwork
from torchchimera.models import ChimeraMagPhasebook
from torchchimera.losses import loss_wa
from torchchimera.metrics import eval_snr
from torchchimera.metrics import eval_si_sdr

class CropRight(torch.nn.Module):
    def __init__(self, waveform_length):
        super(CropRight, self).__init__()
        self.waveform_length = waveform_length
    def forward(self, x):
        if x.shape[-1] < self.waveform_length:
            offset_end = self.waveform_length - x.shape[-1]
            pad_end = torch.zeros(x.shape[0], offset_end)
            x = torch.cat((x, pad_end), dim=-1)
        elif x.shape[-1] > self.waveform_length:
            x = x[:, :self.waveform_length]
        return x

class SimpleTransform(torch.nn.Module):
    def __init__(self, waveform_length, orig_freq=44100, target_freq=16000):
        super(SimpleTransform, self).__init__()
        self.transform = Compose([
            MixTransform([(0, 1, 2), 3, (0, 1, 2, 3)]),
            Resample(orig_freq=orig_freq, new_freq=target_freq),
            CropRight(waveform_length)
        ])
    def forward(self, x):
        return self.transform(x)

"""
input: NxTxFxD tensor
output: NxCxFxT tensor
"""
def from_embedding(embedding, n_channels, n_jobs=-1):
    embedding_dim = embedding.shape[-1]
    labels = sklearn.cluster.KMeans(
        n_clusters=n_channels, n_jobs=n_jobs
    ).fit(
        embedding.reshape(embedding.size // embedding_dim, embedding_dim)
    ).labels_
    mask = np.eye(n_channels)[labels]\
        .reshape(list(embedding.shape[:-1])+[n_channels])\
        .transpose((0, 3, 2, 1))
    return mask

def main():
    dsd_path = '/Volumes/Buffalo 2TB/Datasets/DSD100'
    model_file = 'checkpoint-dc-epoch-1.tar'
    batch_size = 4
    batch_idx = 4
    orig_freq = 44100
    target_freq = 44100
    seconds = 4.5

    n_fft = 2048
    win_length = 2048
    hop_length = 512

    freq_bins, spec_time, _ = torch.stft(
        torch.Tensor(int(seconds * target_freq)), n_fft, hop_length, win_length
    ).shape
    (waveform_length,) = torchaudio.functional.istft(
        torch.Tensor(freq_bins, spec_time, 2), n_fft, hop_length, win_length
    ).shape

    window = torch.sqrt(torch.hann_window(n_fft))
    stft = lambda x: torch.stft(
        x.reshape(x.shape[:-1].numel(), waveform_length),
        n_fft, hop_length, win_length, window=window
    ).reshape(*x.shape[:-1], freq_bins, spec_time, 2)
    istft = lambda X: torchaudio.functional.istft(
        X.reshape(X.shape[:-3].numel(), freq_bins, spec_time, 2),
        n_fft, hop_length, win_length, window=window
    ).reshape(*X.shape[:-3], waveform_length)
    def comp_mul(X, Y):
        (X_re, X_im), (Y_re, Y_im) = X.unbind(-1), Y.unbind(-1)
        return torch.stack((
            X_re * Y_re - X_im * Y_im,
            X_re * Y_im + X_im * Y_re
        ), dim=-1)


    dataset = torch.utils.data.Subset(
        DSD100(dsd_path, 'Test', int(seconds * orig_freq),
               transform=SimpleTransform(
                   waveform_length,
                   orig_freq=orig_freq,
                   target_freq=target_freq
               )
        ),
        indices=list(range(batch_idx*batch_size, (batch_idx+1)*batch_size))
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=8, shuffle=False)

    model = ChimeraMagPhasebook(freq_bins, 2, 20, N=600)
    model.load_state_dict(torch.load(model_file)['model_state_dict'])
    model.eval()
    #misi_layer = MisiNetwork(n_fft, hop_length, win_length, 1)

    batch, _ = next(iter(dataloader))
    s, x = batch[:, :2, :], batch[:, 2, :]
    S, X = stft(s), stft(x)
    S_abs, X_abs = S.norm(p=2, dim=-1), X.norm(p=2, dim=-1)
    Y = torch.eye(2)[
        torch.argmax(S.norm(p=2, dim=-1), dim=1)
        .reshape(batch.shape[0], freq_bins*spec_time)
    ].reshape(batch_size, freq_bins, spec_time, 2)\
    .permute(0, 3, 1, 2).numpy()

    with torch.no_grad():
        embd, (com,), states = model(
            torch.log10(X.norm(p=2, dim=-1).clamp(min=1e-40)),
            outputs=['com'],
            states=None
        )
    embd = embd.reshape(batch.shape[0], -1, freq_bins, 20)
    #'''
    Yhat = from_embedding(embd.numpy(), 2)
    for bi, (y, yhat) in enumerate(zip(Y, Yhat), 1):
        C = y.shape[0]
        vmax = torch.max(X[bi-1])
        plt.subplot(C, 4, 1)
        plt.imshow(X_abs.numpy()[bi-1], aspect='auto', vmin=0, vmax=vmax)
        for ci, (_y, _yhat) in enumerate(zip(y, yhat), 1):
            plt.subplot(C, 4, (ci-1)*4+2)
            plt.imshow(S_abs[bi-1][ci-1], aspect='auto', vmin=0, vmax=vmax)
            plt.subplot(C, 4, (ci-1)*4+3)
            plt.imshow(_yhat * X_abs.numpy()[bi-1], aspect='auto', vmin=0, vmax=vmax)
            #plt.imshow(_yhat, aspect='auto')
            plt.subplot(C, 4, (ci-1)*4+4)
            #plt.imshow(comp_mul(torch.from_numpy(com[bi-1][ci-1]), X[bi-1]).norm(p=2, dim=-1).numpy(), aspect='auto', vmin=0, vmax=vmax)
            plt.imshow((com[bi-1][ci-1].norm(p=2, dim=-1) * X_abs[bi-1]).numpy(), aspect='auto', vmin=0, vmax=vmax)
            #plt.imshow(torch.from_numpy(com[bi-1][ci-1]).norm(p=2, dim=-1).numpy(), aspect='auto')
        plt.show()
    #'''
    Shat = comp_mul(com, X.unsqueeze(1))
    #shat = misi_layer(Shat, x)
    shat = istft(Shat)

    print(loss_wa(shat, s))
    print(eval_snr(shat, s))
    print(eval_si_sdr(shat, s))

    shat = shat.transpose(0, 1).reshape(2, batch_size * waveform_length)
    s = s.transpose(0, 1).reshape(2, batch_size * waveform_length)

    for i_channel, (_s, _shat) in enumerate(zip(s, shat)):
        torchaudio.save(f's_{i_channel}.wav', _s, target_freq)
        torchaudio.save(f'shat_{i_channel}.wav', _shat, target_freq)

if __name__ == '__main__':
    main()
