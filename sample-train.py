#!/usr/bin/env python

import sys
import math
import random
import torch
import torchaudio
from torchaudio.transforms import Vol
from torchaudio.transforms import TimeStretch
#from torchaudio.transforms import Resample
from torchchimera.transforms import Resample # use resampy resample

from torchchimera.datasets import DSD100
from torchchimera.transforms import Compose
from torchchimera.transforms import MixTransform
from torchchimera.transforms import PitchShift
from torchchimera.transforms import RandomCrop
from torchchimera.models import ChimeraMagPhasebook
from torchchimera.submodels import MisiNetwork
from torchchimera.submodels import TrainableMisiNetwork
from torchchimera.losses import loss_mi_tpsa
from torchchimera.losses import loss_dc_whitend
from torchchimera.losses import loss_wa
from torchchimera.losses import loss_csa

def main():
    # parse arguments and set parameters
    batch_size = 32
    orig_freq = 44100
    target_freq = 16000
    seconds = 5
    dataset_path = '/Volumes/Buffalo 2TB/Datasets/DSD100'

    n_fft = 512
    win_length = 512
    hop_length = 128
    freq_bins, spec_time, _ = torch.stft(
        torch.Tensor(seconds * target_freq), n_fft, hop_length, win_length,
        window=torch.hann_window(n_fft)
    ).shape

    # create datasets and dataloaders for train and validation split
    dataset_train = DSD100(dataset_path, 'Dev', seconds * orig_freq)
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size, num_workers=8, shuffle=True)
    dataset_validation = torch.utils.data.Subset(
        DSD100(dataset_path, 'Test', seconds * orig_freq),
        indices=range(500))
    dataloader_validation = torch.utils.data.DataLoader(
        dataset_validation, batch_size=batch_size, num_workers=8)

    # create networks and load from file if needed
    initial_epoch = 16 # start at 1
    train_epoch = 5
    loss_function = 'chimera++' # 'chimera++', 'mask', 'wave'
    input_model = 'model-dc-epoch-15.pth' #'model-dc.pth'
    output_model = 'model-dc-epoch-20.pth'
    input_misi = None
    output_misi = None
    is_misi_trainable = False
    n_misi_layers = 0
    n_input_misi_layers = 0 # default = n_misi_layers

    model = ChimeraMagPhasebook(freq_bins, spec_time, 2, 20, N=600)
    if input_model is not None:
        model.load_state_dict(torch.load(input_model))
    if is_misi_trainable:
        misi = TrainableMisiNetwork(n_fft, n_input_misi_layers)
    else:
        misi = MisiNetwork(n_fft, hop_length, win_length, n_input_misi_layers)
    if input_misi is not None:
        misi.load_state_dict(torch.load(input_misi))
    for _ in range(n_misi_layers - n_input_misi_layers):
        misi.add_layer()

    if not is_misi_trainable:
        params = model.parameters()
    else:
        params = list(model.parameters()) + list(misi.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)

    stft = lambda x: torch.stft(
        x.reshape(x.shape[:-1].numel(), seconds * target_freq),
        n_fft, hop_length, win_length, window=torch.hann_window(n_fft)
    ).reshape(*x.shape[:-1], freq_bins, spec_time, 2)

    for epoch in range(initial_epoch, initial_epoch+train_epoch):
        sum_loss = 0
        total_batch = 0
        last_output_len = 0
        model.train()
        for step, (batch, _) in enumerate(dataloader_train, 1):
            batch = torch.stack(
                [make_random_transform(seconds * target_freq)(b)
                 for b in batch]
            )
            x, s = batch[:, 2, :], batch[:, :2, :]
            X, S = stft(x), stft(s)

            embd, (mag, phasep, com) = model(
                torch.log10(X.norm(p=2, dim=-1).clamp(min=1e-12)),
                outputs=['mag', 'phasep', 'com']
            )

            # compute loss
            if loss_function == 'chimera++':
                S_abs = S.norm(p=2, dim=-1)
                Y = torch.eye(2)[
                    torch.argmax(S_abs + 1e-16*torch.rand_like(S_abs), dim=1)
                    .reshape(batch.shape[0], freq_bins*spec_time)
                ]
                loss = 0.975 * loss_dc_whitend(embd, Y) \
                    + 0.025 * loss_mi_tpsa(mag, X, S, gamma=2.)
            elif loss_function == 'mask':
                loss = 0.5 * loss_mi_tpsa(mag, X, S, gamma=2.) \
                    + 0.5 * loss_csa(com, X, S)
            elif loss_function == 'wave':
                Shat = comp_mul(com, X.unsqueeze(1))
                shat = misi(Shat, x)
                loss = loss_wa(shat, s)

            sum_loss += loss.item()
            total_batch += batch.shape[0]
            ave_loss = sum_loss / total_batch

            # Zero gradients, perform a backward pass, and update the weights.
            loss.backward()
            optimizer.step()
            sum_grad = sum(
                torch.sum(torch.abs(p.grad))
                for n, p in model.named_parameters() if 'blstm' in n and p.grad is not None
            )
            optimizer.zero_grad()

            # Print learning statistics
            curr_output =\
                f'\repoch {epoch} step {step} loss={ave_loss} lstm-grad={sum_grad}'
            sys.stdout.write('\r' + ' ' * last_output_len)
            sys.stdout.write(curr_output)
            sys.stdout.flush()
            last_output_len = len(curr_output)

        model.eval()
        with torch.no_grad():
            sum_val_loss = 0
            total_batch = 0
            for batch, _ in dataloader_validation:
                batch = torch.stack(
                    [SimpleTransform(seconds * target_freq)(b)
                     for b in batch]
                )
                x, s = batch[:, 2, :], batch[:, :2, :]
                X, S = stft(x), stft(s)

                embd, (mag, phasep, com) = model(
                    torch.log10(X.norm(p=2, dim=-1).clamp(min=1e-12)),
                    outputs=['mag', 'phasep', 'com']
                )
                # compute loss
                if loss_function == 'chimera++':
                    S_abs = S.norm(p=2, dim=-1)
                    Y = torch.eye(2)[
                        torch.argmax(S_abs+1e-16*torch.rand_like(S_abs), dim=1)
                        .reshape(batch.shape[0], freq_bins*spec_time)
                    ]
                    loss = 0.975 * loss_dc_whitend(embd, Y) \
                        + 0.025 * loss_mi_tpsa(mag, X, S, gamma=2.)
                elif loss_function == 'mask':
                    loss = 0.5 * loss_mi_tpsa(mag, X, S, gamma=2.) \
                        + 0.5 * loss_csa(com, X, S)
                elif loss_function == 'wave':
                    Shat = comp_mul(com, X.unsqueeze(1))
                    shat = misi(Shat, x)
                    loss = loss_wa(shat, s)
                sum_val_loss += loss.item()
                total_batch += batch.shape[0]
            ave_val_loss = sum_val_loss / total_batch

        curr_output =\
            f'\repoch {epoch} loss={ave_loss}'
        sys.stdout.write('\r' + ' ' * last_output_len)
        sys.stdout.write(f'\repoch {epoch} loss={ave_loss} val={ave_val_loss}\n')

    # save
    if output_model is not None:
        torch.save(model.state_dict(), output_model)
    if is_misi_trainable and output_misi is not None:
        torch.save(misi.state_dict(), output_misi)

def comp_mul(X, Y):
    (X_re, X_im), (Y_re, Y_im) = X.unbind(-1), Y.unbind(-1)
    return torch.stack((
        X_re * Y_re - X_im * Y_im,
        X_re * Y_im + X_im * Y_re
    ), dim=-1)

def make_random_transform(waveform_length, orig_freq=44100, target_freq=16000):
    return CustomTransform(
        waveform_length,
        orig_freq=orig_freq,
        target_freq=target_freq,
        gain=2**random.gauss(0, 0.1),
        stretch_rate=2**random.gauss(0, 0.1),
        shift_rate=2**random.gauss(0, 0.1)
    )

class SimpleTransform(torch.nn.Module):
    def __init__(self, waveform_length, orig_freq=44100, target_freq=16000):
        super(SimpleTransform, self).__init__()
        n_fft = 2048
        hop_length = n_fft // 4
        n_freq = n_fft // 2 + 1
        window = torch.hann_window(n_fft)
        stft = lambda x: torch.stft(x, n_fft, hop_length, window=window)
        istft = lambda x: torchaudio.functional.istft(
            x, n_fft, hop_length, window=window)
        self.transform = Compose([
            MixTransform([(0, 1, 2), 3, (0, 1, 2, 3)]),
            Resample(orig_freq=orig_freq, new_freq=target_freq),
            RandomCrop(waveform_length)
        ])
    def forward(self, x):
        return self.transform(x)

class CustomTransform(torch.nn.Module):
    def __init__(self, waveform_length, orig_freq=44100, target_freq=16000,
                 gain=1.0, stretch_rate=1.0, shift_rate=1.0):
        super(CustomTransform, self).__init__()
        n_fft = 2048
        hop_length = n_fft // 4
        n_freq = n_fft // 2 + 1
        window = torch.hann_window(n_fft)
        stft = lambda x: torch.stft(x, n_fft, hop_length, window=window)
        istft = lambda x: torchaudio.functional.istft(
            x, n_fft, hop_length, window=window)
        self.transform = Compose([
            MixTransform([(0, 1, 2), 3, (0, 1, 2, 3)]),
            Vol(gain),
            stft,
            TimeStretch(hop_length, n_freq, stretch_rate * shift_rate),
            istft,
            Resample(orig_freq=int(orig_freq/shift_rate), new_freq=target_freq),
            RandomCrop(waveform_length)
        ])
    def forward(self, x):
        return self.transform(x)

if __name__ == '__main__':
    main()
