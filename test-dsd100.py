#!/usr/bin/env python

import torch
import torchaudio
from datasets import DSD100, MixTransform

def main():
    batch_size = 5 * 44100
    batch_n = 100
    dataset = DSD100('/Volumes/Buffalo 2TB/Datasets/DSD100', 'Dev', batch_size)
    transform = MixTransform()
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=8, num_workers=8, shuffle=True)
    for batch in dataloader:
        x = transform(batch)
        #torchaudio.save('sample_melody.wav', x[0], 44100)
        #torchaudio.save('sample_vocal.wav', x[1], 44100)

if __name__ == '__main__':
    main()
