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
from torchchimera.metrics import eval_snr
from torchchimera.metrics import eval_si_sdr

class SimpleTransform(torch.nn.Module):
    def __init__(self, orig_freq=44100, target_freq=16000):
        super(SimpleTransform, self).__init__()
        self.transform = Compose([
            MixTransform([(0, 1, 2), 3, (0, 1, 2, 3)]),
            Resample(orig_freq=orig_freq, new_freq=target_freq)
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
    torch.set_printoptions(precision=8)
    dsd_path = '/Volumes/Buffalo 2TB/Datasets/DSD100'
    model_file = 'checkpoint-wa-epoch-5.tar'
    batch_size = 4
    batch_idx = 4
    orig_freq = 44100
    target_freq = 8000
    seconds = 3.2

    n_fft = 256
    win_length = 256
    hop_length = 64

    freq_bins, spec_time, _ = torch.stft(
        torch.Tensor(int(seconds * target_freq)), n_fft, hop_length, win_length
    ).shape

    window = torch.sqrt(torch.hann_window(n_fft))
    stft = lambda x: torch.stft(
        x.reshape(x.shape[:-1].numel(), int(seconds * target_freq)),
        n_fft, hop_length, win_length, window=window
    ).reshape(*x.shape[:-1], freq_bins, spec_time, 2)
    istft = lambda X: torchaudio.functional.istft(
        X.reshape(X.shape[:-3].numel(), freq_bins, spec_time, 2),
        n_fft, hop_length, win_length, window=window
    ).reshape(*X.shape[:-3], int(seconds * target_freq))
    comp_mul = lambda X, Y: torch.stack(
        (X.unbind(-1)[0] * Y.unbind(-1)[0] - X.unbind(-1)[1] * Y.unbind(-1)[1],
         X.unbind(-1)[0] * Y.unbind(-1)[1] + X.unbind(-1)[1] * Y.unbind(-1)[0]),
        dim=-1
    )

    dataset = torch.utils.data.Subset(
        DSD100(dsd_path, 'Test', int(seconds * orig_freq),
               transform=SimpleTransform(orig_freq, target_freq)),
        indices=list(range(batch_idx*batch_size, (batch_idx+1)*batch_size))
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=8, shuffle=False)

    model = ChimeraMagPhasebook(freq_bins, 2, 20, N=600)
    model.load_state_dict(torch.load(model_file)['model_state_dict'])
    model.eval()
    misi_layer = MisiNetwork(n_fft, hop_length, win_length, 1)

    batch, _ = next(iter(dataloader))
    s1, s2, x = batch.unbind(1)
    s = torch.stack((s1, s2), dim=1)
    S, X = stft(s), stft(x)
    X_abs = torch.sqrt(torch.sum(X**2, dim=-1))
    X_phase = X / X_abs.clamp(min=1e-24).unsqueeze(-1)
    S_abs = S.norm(p=2, dim=-1)
    Y = torch.eye(2)[
        torch.argmax(S.norm(p=2, dim=-1), dim=1)
        .reshape(batch.shape[0], freq_bins*spec_time)
    ].reshape(batch_size, freq_bins, spec_time, 2)\
    .permute(0, 3, 1, 2).numpy()

    states = None
    embd = np.zeros((0, spec_time, freq_bins, 20))
    com = np.zeros((0, 2, freq_bins, spec_time, 2))
    for x_abs in X_abs:
        with torch.no_grad():
            states = None
            _embd, (_com,), states = model(
                torch.log10(x_abs.clamp(min=1e-24)).unsqueeze(0),
                outputs=['com'], states=states)
        _embd = _embd.detach()\
                     .reshape(1, spec_time, freq_bins, 20)\
                     .numpy()
        _com = _com.detach().numpy()
        embd = np.concatenate((embd, _embd))
        com = np.concatenate((com, _com))
    '''
    Yhat = from_embedding(embd, 2)
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
            plt.imshow(torch.from_numpy(com[bi-1][ci-1]).norm(p=2, dim=-1).numpy() * X_abs.numpy()[bi-1], aspect='auto', vmin=0, vmax=vmax)
            #plt.imshow(torch.from_numpy(com[bi-1][ci-1]).norm(p=2, dim=-1).numpy(), aspect='auto')
        #plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
        #cax = plt.axes([0.85, 0.1, 0.075, 0.8])
        #plt.colorbar(cax=cax)
        plt.tight_layout()
        plt.show()
    '''
    #com = com.detach()
    phase_comp = lambda X: torch.atan2(*X.split(1, dim=-1)[::-1]).squeeze(-1)
    S_abs = S.norm(p=2, dim=-1)
    S_phase = phase_comp(S)
    X_abs = X.norm(p=2, dim=-1).unsqueeze(1)
    X_phase = phase_comp(X.unsqueeze(1))
    spectrum = torch.min(
        input=torch.max(
            input=S_abs * torch.cos(S_phase - X_phase),
            other=torch.zeros_like(S_abs)
        ),
        other=2*X_abs
    )
    Shat = comp_mul(torch.from_numpy(com), X.unsqueeze(1))
    #shat = misi_layer(Shat, x)
    shat = istft(Shat)
    s = torchaudio.functional.istft(
        S.reshape(batch_size*2, freq_bins, spec_time, 2),
        n_fft, hop_length, win_length,
        window=torch.hann_window(n_fft)
    ).reshape(batch_size, 2, int(seconds * target_freq))

    print(eval_snr(shat, s))
    print(eval_si_sdr(shat, s))

    shat = shat.transpose(0, 1).reshape(2, batch_size * int(seconds * target_freq))
    s = s.transpose(0, 1).reshape(2, batch_size * int(seconds * target_freq))

    for i_channel, (_s, _shat) in enumerate(zip(s, shat)):
        torchaudio.save(f's_{i_channel}.wav', _s, target_freq)
        torchaudio.save(f'shat_{i_channel}.wav', _shat, target_freq)

if __name__ == '__main__':
    main()
