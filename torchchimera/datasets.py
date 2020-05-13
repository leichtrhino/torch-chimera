
import os
import math
import bisect

import torch
import torchaudio

class DSD100(torch.utils.data.Dataset):
    def __init__(self, root_dir, split, waveform_length,
                 sources=('bass', 'drums', 'other', 'vocals'),
                 transform=None):
        self.sources = sources
        self.waveform_length = waveform_length
        self.offsets = [0]
        self.rates = []
        self.parent_dir = os.path.join(root_dir, 'Sources', split)
        self.transform = transform
        self.source_dirs = sorted(filter(
            lambda d: os.path.isdir(d),
            map(
                lambda d: os.path.join(self.parent_dir, d),
                os.listdir(self.parent_dir)
            )
        ))
        for d in self.source_dirs:
            si, _ = torchaudio.info(os.path.join(d, 'bass.wav'))
            self.offsets.append(
                self.offsets[-1] + math.ceil(si.length / si.channels / self.waveform_length)
            )
            self.rates.append(si.rate)

    def __len__(self):
        return self.offsets[-1]

    def _get_single_item(self, idx):
        audio_idx = bisect.bisect(self.offsets, idx) - 1
        offset_idx = idx - self.offsets[audio_idx]
        x = torch.stack([
            torchaudio.load(
                os.path.join(self.source_dirs[audio_idx], s+'.wav'),
                offset=offset_idx * self.waveform_length,
                num_frames=self.waveform_length
            )[0]
            for s in self.sources
        ]).mean(axis=1)
        if x.shape[-1] < self.waveform_length:
            x = torch.cat((
                x,
                torch.zeros(
                    len(self.sources),
                    self.waveform_length-x.shape[-1]
                )
            ), dim=-1)
        return x, self.rates[audio_idx]

    def __getitem__(self, idx):
        if type(idx) == int:
            waveform, rate = self._get_single_item(idx)
            if callable(self.transform):
                waveform = self.transform(waveform)
            return waveform, rate
        if torch.is_tensor(idx):
            idx = idx.tolist()
        waveforms, rates = zip(*[self._get_single_item(i) for i in idx])
        waveforms = torch.stack(waveforms)
        if callable(self.transform):
            waveforms = self.transform(waveforms)
        return waveforms, rates
