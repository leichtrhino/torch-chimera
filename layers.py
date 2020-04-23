
import torch
import torchaudio

class MisiLayer(torch.nn.Module):
    def __init__(self, n_fft, hop_length, win_length):
        super(MisiLayer, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
    # input: (batch_size, n_channels, freq_bins, spec_time)
    # input: (batch_size, n_channels, freq_bins, spec_time, 2)
    # input: (batch_size, waveform_length)
    # output: (batch_size, n_channels, freq_bins, spec_time, 2)
    def forward(self, mag, phase, mixture):
        batch_size, n_channels, freq_bins, spec_time = mag.shape
        waveform_length = mixture.shape[-1]
        s = torchaudio.functional.istft(
            (mag.unsqueeze(-1) * phase).reshape(batch_size*n_channels, freq_bins, spec_time, 2),
            self.n_fft, self.hop_length, self.win_length
        ).reshape(batch_size, n_channels, waveform_length)
        delta = mixture - torch.sum(s, dim=1)
        Shat = torch.stft(
            (s + delta.unsqueeze(1) / n_channels).reshape(batch_size*n_channels, waveform_length),
            self.n_fft, self.hop_length, self.win_length
        ).reshape(batch_size, n_channels, freq_bins, spec_time, 2)
        return Shat / torch.sqrt(torch.sum(Shat**2, dim=-1).clamp(min=1e-9)).unsqueeze(-1)
