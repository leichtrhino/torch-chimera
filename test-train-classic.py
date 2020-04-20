#!/usr/bin/env python

import sys
import math
import torch
import torchaudio
from datasets import DSD100, MixTransform
from models import ClassicChimera
from losses import loss_mi_msa, loss_dc

def main():
    batch_size = 32
    orig_freq = 44100
    target_freq = 16000
    seconds = 5

    n_fft = 512
    win_length = 512
    hop_length = 128
    freq_bins, spec_time, _ = torch.stft(
        torch.Tensor(seconds * target_freq), n_fft, hop_length, win_length
    ).shape

    dataset = DSD100(
        '/Volumes/Buffalo 2TB/Datasets/DSD100', 'Dev', seconds * orig_freq)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=8, shuffle=True)
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

    model = ClassicChimera(freq_bins, spec_time, 2, 20)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(10):
        sum_loss = 0
        last_output_len = 0
        for step, batch in enumerate(dataloader):
            batch = transform(batch)
            X = batch[:, 2, :, :, :]
            S = batch[:, :2, :, :, :]
            X_abs = torch.sqrt(torch.sum(X**2, dim=-1))
            S_abs = torch.sqrt(torch.sum(S**2, dim=-1))
            Y = torch.eye(2)[
                torch.argmax(S_abs, dim=1)
                .reshape(batch.shape[0], freq_bins*spec_time)
            ]
            embd, mask = model(torch.log10(X_abs.clamp(min=1e-9)))

            # Compute loss
            loss = loss_dc(embd, Y) + loss_mi_msa(mask, X, S)
            sum_loss += loss.item() * batch.shape[0]
            ave_loss = sum_loss / (batch.shape[0]*(step+1))

            # Zero gradients, perform a backward pass, and update the weights.
            loss.backward()
            optimizer.step()
            sum_grad = sum(
                torch.sum(torch.abs(p.grad))
                for p in model.parameters() if p.grad is not None
            )
            optimizer.zero_grad()

            # Print learning statistics
            curr_output =\
                f'\repoch {epoch} step {step} loss={ave_loss} grad={sum_grad}'
            sys.stdout.write('\r' + ' ' * last_output_len)
            sys.stdout.write(curr_output)
            sys.stdout.flush()
            last_output_len = len(curr_output)

        curr_output =\
            f'\repoch {epoch} loss={ave_loss}'
        sys.stdout.write('\r' + ' ' * last_output_len)
        sys.stdout.write(f'\repoch {epoch} loss={ave_loss}\n')

        torch.save(
            model.state_dict(),
            f'model_epoch{epoch}.pth'
        )


if __name__ == '__main__':
    main()
