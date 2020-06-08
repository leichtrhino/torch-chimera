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

import matplotlib.pyplot as plt

def label_matrix(S):
    batch_size, n_channels, freq_bins, spec_time, _ = S.shape
    S_abs = S.norm(p=2, dim=-1)
    p = S_abs.transpose(1, 3).reshape(batch_size, spec_time*freq_bins, n_channels).softmax(dim=-1).cumsum(dim=-1)
    r = torch.rand(batch_size, spec_time * freq_bins)
    k = torch.eye(n_channels)[torch.argmin(torch.where(r.unsqueeze(-1) <= p, p, torch.ones_like(p)), dim=-1)]
    return k

def weight_matrix(X):
    batch_size, freq_bins, spec_time, _ = X.shape
    X_abs = X.norm(p=2, dim=-1)
    weight = X_abs.transpose(1, 2).reshape(batch_size, spec_time*freq_bins)\
        / X_abs.sum(dim=(1, 2)).clamp(min=1e-16).unsqueeze(-1)
    return weight

def main():
    # parse arguments and set parameters
    batch_size = 8
    validation_batch_size = 8
    orig_freq = 44100
    target_freq = 16000
    seconds = 3
    dataset_path = '/Volumes/Buffalo 2TB/Datasets/DSD100'

    n_fft = 512
    win_length = 512
    hop_length = 128
    freq_bins, spec_time, _ = torch.stft(
        torch.Tensor(seconds * target_freq), n_fft, hop_length, win_length,
        window=torch.hann_window(n_fft)
    ).shape

    # create datasets and dataloaders for train and validation split
    dataset_train = DSD100(dataset_path, 'Dev', None)
    dataset_train.transform = RandomTransform(
        int(dataset_train.get_max_length() / orig_freq * target_freq))
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size, num_workers=2, shuffle=True)

    dataset_validation = DSD100(dataset_path, 'Test', None)
    dataset_validation.transform = SimpleTransform(
        int(dataset_validation.get_max_length() / orig_freq * target_freq))
    dataset_validation = torch.utils.data.Subset(dataset_validation, range(8))
    dataloader_validation = torch.utils.data.DataLoader(
        dataset_validation, batch_size=validation_batch_size,
        num_workers=2, shuffle=False)

    def generate_batch(dataloader, waveform_length):
        for (B, _) in dataloader:
            batch_size = math.ceil(B.shape[-1] / waveform_length)
            for batch_i in range(batch_size):
                batch_end = batch_i == batch_size - 1
                end = (batch_i + 1) * waveform_length
                if end <= B.shape[-1]:
                    yield B[:, :, batch_i*waveform_length:end], batch_end
                else:
                    yield torch.cat((
                        B[:, :, batch_i*waveform_length:],
                        torch.zeros(B.shape[0], B.shape[1], end-B.shape[2])
                    ), dim=-1), batch_end

    # create networks and load from file if needed
    initial_epoch = 9 # start at 1
    train_epoch = 1
    loss_function = 'chimera++' # 'chimera++', 'mask', 'wave'
    input_model = f'model-dc-epoch-{initial_epoch-1}.pth'\
        if initial_epoch > 1 else None
    output_model = f'model-dc-epoch-{initial_epoch+train_epoch-1}.pth'
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

    def forward(model, batch, states=None):
        stft = lambda x: torch.stft(
            x.reshape(x.shape[:-1].numel(), seconds * target_freq),
            n_fft, hop_length, win_length, window=torch.hann_window(n_fft)
        ).reshape(*x.shape[:-1], freq_bins, spec_time, 2)

        x, s = batch[:, 2, :], batch[:, :2, :]
        X, S = stft(x), stft(s)

        embd, (mag, phasep, com), states = model(
            torch.log10(X.norm(p=2, dim=-1).clamp(min=1e-24)),
            outputs=['mag', 'phasep', 'com'],
            states=states
        )
        states = tuple(state.detach() for state in states)

        # compute loss
        if loss_function == 'chimera++':
            Y = label_matrix(S)
            weight = weight_matrix(X)
            loss = 0.975 * loss_dc_whitend(embd, Y, weight) \
                + 0.025 * loss_mi_tpsa(mag, X, S, gamma=2.)
        elif loss_function == 'mask':
            loss = 0.5 * loss_mi_tpsa(mag, X, S, gamma=2.) \
                + 0.5 * loss_csa(com, X, S)
        elif loss_function == 'wave':
            Shat = comp_mul(com, X.unsqueeze(1))
            shat = misi(Shat, x)
            loss = loss_wa(shat, s)
        return loss, states

    for epoch in range(initial_epoch, initial_epoch+train_epoch):
        sum_loss = 0
        total_batch = 0
        ave_loss = 0
        last_output_len = 0
        states = None
        model.train()
        for step, (batch, batch_end) in enumerate(
                generate_batch(dataloader_train, target_freq * seconds), 1):
            loss, states = forward(model, batch, states)
            if batch_end:
                states = None
            sum_loss += loss.item()
            total_batch += batch.shape[0]
            ave_loss = sum_loss / total_batch

            # Zero gradients, perform a backward pass, and update the weights.
            loss.backward()
            optimizer.step()
            sum_grad = sum(
                torch.sum(torch.abs(p.grad))
                for n, p in model.named_parameters()
                if 'blstm' in n and p.grad is not None
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
            states = None
            for (batch, batch_end) in generate_batch(
                    dataloader_validation, target_freq * seconds):
                loss, states = forward(model, batch, states)
                if batch_end:
                    states = None
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
        stretch_rate=2**random.gauss(0, 0.01),
        shift_rate=2**random.gauss(0, 0.01)
    )

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
            Resample(orig_freq=orig_freq/shift_rate, new_freq=target_freq),
            RandomCrop(waveform_length)
        ])
    def forward(self, x):
        return self.transform(x)

class RandomTransform(torch.nn.Module):
    def __init__(self, waveform_length, orig_freq=44100, target_freq=16000):
        super(RandomTransform, self).__init__()
        self.waveform_length = waveform_length
        self.orig_freq = orig_freq
        self.target_freq = target_freq
    def forward(self, x):
        return CustomTransform(
            self.waveform_length,
            self.orig_freq,
            self.target_freq,
            gain=2**random.gauss(0, 0.1),
            stretch_rate=2**random.gauss(0, 0.01),
            shift_rate=2**random.gauss(0, 0.01)
        )(x)

if __name__ == '__main__':
    main()
