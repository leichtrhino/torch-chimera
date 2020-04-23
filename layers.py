
import torch
import torchaudio

class MisiLayer(torch.nn.Module):
    def __init__(self, n_fft, hop_length, win_length, layer_num=1):
        super(MisiLayer, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.layer_num = layer_num
    # input: (batch_size, n_channels, freq_bins, spec_time)
    # input: (batch_size, n_channels, freq_bins, spec_time, 2)
    # input: (batch_size, waveform_length)
    # output: (batch_size, n_channels, freq_bins, spec_time, 2)
    def forward(self, mag, phase, mixture):
        batch_size, n_channels, freq_bins, spec_time = mag.shape
        waveform_length = mixture.shape[-1]
        for _ in range(self.layer_num):
            S = mag.unsqueeze(-1) * phase
            s = torchaudio.functional.istft(
                S.reshape(batch_size*n_channels, freq_bins, spec_time, 2),
                self.n_fft, self.hop_length, self.win_length
            ).reshape(batch_size, n_channels, waveform_length)
            delta = mixture - torch.sum(s, dim=1)
            shat = s + delta.unsqueeze(1) / n_channels
            Shat = torch.stft(
                shat.reshape(batch_size*n_channels, waveform_length),
                self.n_fft, self.hop_length, self.win_length
            ).reshape(batch_size, n_channels, freq_bins, spec_time, 2)
            Shat_mag = torch.sqrt(torch.sum(Shat**2, dim=-1).clamp(min=1e-9))
            phase = Shat / Shat_mag.unsqueeze(-1)
        return phase
