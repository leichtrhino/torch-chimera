
import torch
import torchaudio

class MisiLayer(torch.nn.Module):
    def __init__(self, n_fft, hop_length, win_length, layer_num=1):
        super(MisiLayer, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.layer_num = layer_num
    # input: (batch_size, n_channels, freq_bins, spec_time, 2)
    # input: (batch_size, waveform_length)
    # output: (batch_size, n_channels, freq_bins, spec_time, 2)
    def forward(self, Shat, mixture):
        batch_size, n_channels, freq_bins, spec_time, _ = Shat.shape
        waveform_length = mixture.shape[-1]
        stft = lambda x: torch.stft(
            x.reshape(x.shape[:-1].numel(), waveform_length),
            self.n_fft, self.hop_length, self.win_length
        ).reshape(*x.shape[:-1], freq_bins, spec_time, 2)
        istft = lambda X: torchaudio.functional.istft(
            X.reshape(X.shape[:-3].numel(), freq_bins, spec_time, 2),
            self.n_fft, self.hop_length, self.win_length
        ).reshape(*X.shape[:-3], waveform_length)

        mag = torch.sqrt(torch.sum(Shat**2, dim=-1).clamp(min=1e-24))
        phasehat = Shat / mag.unsqueeze(-1)
        shat = istft(Shat)
        for _ in range(self.layer_num):
            delta = mixture - torch.sum(shat, dim=1)
            shat_tmp = shat + delta.unsqueeze(1) / n_channels
            Shat = stft(shat_tmp)
            _mag = torch.sqrt(torch.sum(Shat**2, dim=-1).clamp(min=1e-24))
            phasehat = Shat / _mag.unsqueeze(-1)
            shat = istft(mag.unsqueeze(-1) * phasehat)
        return phasehat, shat
