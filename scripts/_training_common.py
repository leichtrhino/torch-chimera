
import os
import sys
import math

import torch
import torchaudio

try:
    import torchchimera
except:
    # attempts to import local module
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    import torchchimera
from torchchimera.losses import permutation_free
from torchchimera.losses import loss_mi_tpsa
from torchchimera.losses import loss_dc_deep_lda
from torchchimera.losses import loss_wa
from torchchimera import metrics

class StftSetting(object):
    def __init__(self, n_fft, hop_length=None, win_length=None, window=None):
        self.n_fft = n_fft
        if hop_length is None:
            self.hop_length = n_fft // 4
        else:
            self.hop_length = hop_length
        if win_length is None:
            self.win_length = n_fft
        else:
            self.win_length = win_length
        self.window = window

class Stft(torch.nn.Module):
    def __init__(self, stft_setting):
        super(Stft, self).__init__()
        self.stft_setting = stft_setting

    def forward(self, x):
        waveform_length = x.shape[-1]
        y = torch.stft(
            x.reshape(x.shape[:-1].numel(), waveform_length),
            self.stft_setting.n_fft,
            self.stft_setting.hop_length,
            self.stft_setting.win_length,
            window=self.stft_setting.window
        )
        _, freq, time, _ = y.shape
        return y.reshape(*x.shape[:-1], freq, time, 2)

class Istft(torch.nn.Module):
    def __init__(self, stft_setting):
        super(Istft, self).__init__()
        self.stft_setting = stft_setting

    def forward(self, x):
        freq, time = x.shape[-3], x.shape[-2]
        y = torchaudio.functional.istft(
            x.reshape(x.shape[:-3].numel(), freq, time, 2),
            self.stft_setting.n_fft,
            self.stft_setting.hop_length,
            self.stft_setting.win_length,
            window=self.stft_setting.window
        )
        waveform_length = y.shape[-1]
        return y.reshape(*x.shape[:-3], waveform_length)

class AdaptedChimeraMagPhasebook(torch.nn.Module):
    def __init__(self, chimera, stft_setting):
        super(AdaptedChimeraMagPhasebook, self).__init__()
        self.chimera = chimera
        self.stft_setting = stft_setting

    def forward(self, x, states=None):
        def comp_mul(X, Y):
            (X_re, X_im), (Y_re, Y_im) = X.unbind(-1), Y.unbind(-1)
            return torch.stack((
                X_re * Y_re - X_im * Y_im,
                X_re * Y_im + X_im * Y_re
            ), dim=-1)
        stft = Stft(self.stft_setting)
        istft = Istft(self.stft_setting)

        X = stft(x)
        embd, (mag, com), out_status = self.chimera(
            torch.log10(X.norm(p=2, dim=-1).clamp(min=1e-40)),
            states=states, outputs=['mag', 'com']
        )
        shat = istft(comp_mul(com, X.unsqueeze(1)))
        return embd, mag, shat, out_status

class AdaptedChimeraMagPhasebookWithMisi(torch.nn.Module):
    def __init__(self, chimera, misi, stft_setting):
        super(AdaptedChimeraMagPhasebookWithMisi, self).__init__()
        self.chimera = chimera
        self.misi = misi
        self.stft_setting = stft_setting

    def forward(self, x, states=None):
        X = Stft(self.stft_setting)(x)
        embd, (mag, com), out_status = self.chimera(
            torch.log10(X.norm(p=2, dim=-1).clamp(min=1e-40)),
            states=states, outputs=['mag', 'com']
        )
        shat = self.misi(com, x)
        return embd, mag, shat, out_status

def dc_label_matrix(S):
    batch_size, n_channels, freq_bins, spec_time, _ = S.shape
    S_abs = S.norm(p=2, dim=-1)
    p = S_abs.transpose(1, 3).reshape(batch_size, spec_time*freq_bins, n_channels).softmax(dim=-1).cumsum(dim=-1)
    r = torch.rand(batch_size, spec_time * freq_bins, device=S.device)
    k = torch.eye(n_channels, device=S.device)[torch.argmin(torch.where(r.unsqueeze(-1) <= p, p, torch.ones_like(p)), dim=-1)]
    return k

def dc_weight_matrix(X):
    batch_size, freq_bins, spec_time, _ = X.shape
    X_abs = X.norm(p=2, dim=-1)
    weight = X_abs.transpose(1, 2).reshape(batch_size, spec_time*freq_bins)\
        / X_abs.sum(dim=(1, 2)).clamp(min=1e-16).unsqueeze(-1)
    return weight

def compute_loss(s, y_pred, stft_setting,
                 loss_function='chimera++', is_permutation_free=False):
    stft = Stft(stft_setting)
    x = s.sum(dim=1)
    S, X = stft(s), stft(x)

    embd, mag, shat, _ = y_pred
    if loss_function == 'chimera++':
        Y = dc_label_matrix(S)
        weight = dc_weight_matrix(X)
        alpha = 0.975
        loss_dc = alpha * loss_dc_deep_lda(embd, Y, weight)
        if is_permutation_free:
            loss_mi = (1-alpha) * permutation_free(loss_mi_tpsa)(mag, X, S, gamma=2.)
        else:
            loss_mi = (1-alpha) * loss_mi_tpsa(mag, X, S, gamma=2.)
        loss = loss_dc + loss_mi
    elif loss_function == 'wave-approximation':
        waveform_length = min(s.shape[-1], shat.shape[-1])
        s = s[:, :, :waveform_length]
        shat = shat[:, :, :waveform_length]
        if is_permutation_free:
            loss = permutation_free(loss_wa)(shat, s)
        else:
            loss = loss_wa(shat, s)
    elif loss_function == 'si-sdr':
        waveform_length = min(s.shape[-1], shat.shape[-1])
        s = torch.clamp(s[:, :, :waveform_length], min=1e-120)
        shat = torch.clamp(shat[:, :, :waveform_length], min=1e-120)
        if is_permutation_free:
            score = permutation_free(lambda x, y: -1*torch.sum(metrics.eval_si_sdr(x, y))) \
            (shat, s)
        else:
            score = torch.sum(metrics.eval_si_sdr(shat, s))
        loss = -1 * score / (s.shape[0] * s.shape[1])
    else:
        loss = None
    return loss

def exclude_silence(s, stft_setting, cutoff_rms):
    stft = Stft(stft_setting)
    S = stft(s.squeeze(0))
    S_pow = torch.sum(stft(s.squeeze(0))**2, dim=-1)
    rms = 10 * torch.log10(torch.mean(S_pow, dim=1))
    is_no_silence = torch.all(rms > cutoff_rms, dim=0)
    S = S[:, :, is_no_silence, :]
    if S.shape[2] < 4: # 4: n_fft // hop_length
        return None
    istft = Istft(stft_setting)
    s = istft(S).unsqueeze(0)
    return s
