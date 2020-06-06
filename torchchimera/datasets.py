
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
        self.max_length = 0 if waveform_length is None else waveform_length
        for d in self.source_dirs:
            si, _ = torchaudio.info(os.path.join(d, 'bass.wav'))
            if waveform_length is None:
                self.max_length = max(self.max_length, si.length // si.channels)
            offset_diff = 1 if self.waveform_length is None else\
                math.ceil(si.length / si.channels / self.waveform_length)
            self.offsets.append(self.offsets[-1] + offset_diff)
            self.rates.append(si.rate)

    def __len__(self):
        return self.offsets[-1]

    def get_max_length(self):
        return self.max_length

    def _get_single_item(self, idx):
        audio_idx = bisect.bisect(self.offsets, idx) - 1
        offset_idx = idx - self.offsets[audio_idx]
        offset = 0 if self.waveform_length is None else\
            offset_idx * self.waveform_length
        num_frames = 0 if self.waveform_length is None else\
            self.waveform_length
        x = torch.stack([
            torchaudio.load(
                os.path.join(self.source_dirs[audio_idx], s+'.wav'),
                offset=offset, num_frames=num_frames
            )[0]
            for s in self.sources
        ]).mean(axis=1)
        if self.waveform_length is not None and\
           x.shape[-1] < self.waveform_length:
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
