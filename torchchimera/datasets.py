
import os
import math
import bisect
import pathlib
import random

import torch
import torchaudio
import resampy

class Folder(torch.utils.data.Dataset):
    def __init__(self, root_dir, sr, duration=None, transform=None):
        self.sr = sr
        self.duration = duration
        self.transform = transform
        self.offsets = [0]
        self.rates = []

        self.paths = sorted(list(pathlib.Path(root_dir).glob('**/*.wav')))

        for p in self.paths:
            metadata = torchaudio.info(str(p))
            self.rates.append(metadata.sample_rate)
            if self.duration is None:
                self.offsets.append(self.offsets[-1] + 1)
                continue
            n_segments = math.floor(
                metadata.num_frames / metadata.sample_rate / self.duration)
            self.offsets.append(self.offsets[-1] + n_segments)

    def __len__(self):
        return self.offsets[-1]

    def __getitem__(self, idx):
        audio_idx = bisect.bisect(self.offsets, idx) - 1
        offset_idx = idx - self.offsets[audio_idx]
        if self.duration is None:
            offset = 0
            num_frames = 0
        else:
            offset = offset_idx * int(self.duration * self.rates[audio_idx])
            num_frames = int(self.rates[audio_idx] * self.duration)
        x, _ = torchaudio.load(
            str(self.paths[audio_idx]), frame_offset=offset, num_frames=num_frames
        )
        x = x.mean(dim=0)
        if x.shape[-1] * self.sr / self.rates[audio_idx] < 1:
            x = torch.zeros((
                *x.shape[:-1], math.ceil(self.rates[audio_idx] / self.sr)
            ))
        x = torch.Tensor(resampy.resample(
            x.numpy(), self.rates[audio_idx], self.sr, axis=-1
        ))
        if self.duration is not None:
            out_length = int(self.sr * self.duration)
            if x.shape[-1] > out_length:
                x = x[:self.out_length]
            if x.shape[-1] < out_length:
                x = torch.cat((x, torch.zeros(out_length-x.shape[-1])))
        if self.transform is not None:
            x = self.transform(x)
        return x

class FolderTuple(torch.utils.data.Dataset):
    def __init__(self, root_dirs, sr, duration=None, transform=None):
        self.folders = [
            Folder(d, sr, duration, transform) for d in root_dirs
        ]
        self.unshuffle()

    def __len__(self):
        return min(len(f) for f in self.folders)

    def shuffle(self):
        idx_list = [
            list(range(len(f))) for f in self.folders
        ]
        for il in idx_list:
            random.shuffle(il)
        self.idx_list = list(zip(*idx_list))

    def unshuffle(self):
        self.idx_list = [
            tuple(i for _ in self.folders) for i in range(len(self))
        ]

    def __getitem__(self, idx):
        x_list = [f[i] for f, i in zip(self.folders, self.idx_list[idx])]
        segment_length = min(x.shape[-1] for x in x_list)
        return torch.stack([x[:segment_length] for x in x_list])

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
