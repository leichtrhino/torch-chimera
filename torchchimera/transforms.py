
import math
import random
import torch
import torchaudio
import resampy

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

class Resample(torch.nn.Module):
    def __init__(self, orig_freq, new_freq):
        super(Resample, self).__init__()
        self.orig_freq = orig_freq
        self.new_freq = new_freq

    def forward(self, x):
        return torch.Tensor(resampy.resample(
            x.numpy(), self.orig_freq, self.new_freq, axis=-1))

class MixTransform(torch.nn.Module):
    def __init__(self, source_lists=[(0, 1, 2), 3], source_coeffs=None):
        super(MixTransform, self).__init__()
        self.source_lists = [
            torch.Tensor(s if hasattr(s, '__len__') else [s]).long()
            for s in source_lists
        ]
        if source_coeffs is None:
            self.source_coeffs = [
                torch.ones(len(s)) if hasattr(s, '__len__') else
                torch.ones(1) for s in source_lists
            ]
        else:
            self.source_coeffs = source_coeffs

    def forward(self, sample):
        # weighted sum along dim=-2 (dim=-1 are waveforms)
        return torch.stack([
            torch.sum(
                sc.unsqueeze(-1) * sample.index_select(dim=-2, index=sl),
                dim=-2
            )
            for sl, sc in zip(self.source_lists, self.source_coeffs)
        ], dim=-2)

class RandomScale(torch.nn.Module):
    def __init__(self):
        super(RandomScale, self).__init__()

    def forward(self, x):
        source_amplitude = torch.max(x.abs(), dim=-1, keepdim=True)[0]
        scale = 10 ** (torch.rand(
            (*x.shape[:-1], 1), device=x.device) * -10 / 10)
        return x * scale\
            / torch.where(
                source_amplitude > .1,
                source_amplitude,
                torch.ones_like(source_amplitude)
            )

class PitchShift(torch.nn.Module):
    def __init__(self, sampling_rate, shift_rate, n_fft=512):
        super(PitchShift, self).__init__()
        self.stft = lambda x: torch.stft(
            x, n_fft, hop_length=n_fft//4,
            window=torch.hann_window(n_fft)
        )
        self.istft = lambda x: torch.istft(
            x, n_fft, hop_length=n_fft//4,
            window=torch.hann_window(n_fft)
        )
        self.time_stretch = torchaudio.transforms.TimeStretch(
            hop_length=n_fft//4,
            n_freq=(n_fft//2)+1,
            fixed_rate=shift_rate
        )
        self.resample = Resample(
            orig_freq=sampling_rate/shift_rate,
            new_freq=sampling_rate
        )

    def forward(self, x):
        spec_orig = self.stft(x)
        spec_target = self.time_stretch(spec_orig)
        x_target = self.istft(spec_target)
        return self.resample(x_target)

class RandomPadOrCrop(torch.nn.Module):
    def __init__(self, waveform_length):
        super(RandomPadOrCrop, self).__init__()
        self.waveform_length = waveform_length

    def forward(self, x):
        if x.shape[-1] < self.waveform_length:
            if len(x.shape) == 1:
                x = x.unsqueeze(0)

            offset = random.randint(0, self.waveform_length-x.shape[-1])
            # pad reflect
            pad_begin = x.repeat((1, math.ceil(offset / x.shape[-1])))[
                ..., math.ceil(offset / x.shape[-1]) * x.shape[-1] - offset:
            ]
            offset_end = self.waveform_length-offset-x.shape[-1]
            pad_end = x.repeat((1, math.ceil(offset_end / x.shape[-1])))[
                ..., :offset_end
            ]
            x = torch.cat((pad_begin, x, pad_end), dim=-1).squeeze()
        elif x.shape[-1] > self.waveform_length:
            offset = random.randint(0, x.shape[-1]-self.waveform_length)
            x = x[..., offset:offset+self.waveform_length]
        return x


def build_transform(orig_freq, new_freq, stft_setting, waveform_length):
    time_stretch_rate = random.uniform(0.7, 1.3)
    pitch_shift_rate = random.uniform(0.7, 1.3)
    transforms = [
        PitchShift(orig_freq, pitch_shift_rate, n_fft=stft_setting.n_fft),
        lambda x: torch.stft(
            x,
            stft_setting.n_fft,
            stft_setting.hop_length,
            stft_setting.win_length,
        ),
        torchaudio.transforms.TimeStretch(
            hop_length=stft_setting.hop_length,
            fixed_rate=time_stretch_rate,
            n_freq=stft_setting.n_fft//2+1
        ),
        lambda x: torch.istft(
            x,
            stft_setting.n_fft,
            stft_setting.hop_length,
            stft_setting.win_length,
        ),
        Resample(float(orig_freq), float(new_freq)),
        RandomScale(),
    ]
    if waveform_length is not None:
        transforms.append(RandomPadOrCrop(waveform_length))
    return Compose(transforms)

class CombinedRandomTransform(torch.nn.Module):
    def __init__(self, orig_sr, target_sr, stft_setting, waveform_length):
        super(CombinedRandomTransform, self).__init__()
        self.orig_sr = orig_sr
        self.target_sr = target_sr
        self.stft_setting = stft_setting
        self.waveform_length = waveform_length
    def forward(self, x):
        return build_transform(
            self.orig_sr, self.target_sr,
            self.stft_setting, self.waveform_length
        )(x)
