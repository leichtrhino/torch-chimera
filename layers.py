
import torch
import torchaudio

class MisiLayer(torch.nn.Module):
    def __init__(self, n_fft, hop_length, win_length):
        super(MisiLayer, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
    # input: (batch_size * n_channels, freq_bins, spec_time, 2)
    # input: (batch_size * n_channels, freq_bins, spec_time, 1)
    # input: (batch_size, waveform_length)
    # output: (batch_size * n_channels, freq_bins, spec_time, 2)
    def forward(self, Shat, Shatmag, mixture):
        batch_size, waveform_length = mixture.shape
        n_channels, freq_bins, spec_time, _ = Shat.shape
        n_channels //= batch_size
        waveform_length = mixture.shape[-1]

        stft = lambda x: torch.stft(
            x, self.n_fft, self.hop_length, self.win_length,
            window=torch.hann_window(self.n_fft)
        )
        istft = lambda X: torchaudio.functional.istft(
            X, self.n_fft, self.hop_length, self.win_length,
            window=torch.hann_window(self.n_fft)
        )

        shat = istft(Shat)
        delta = mixture - torch.sum(
            shat.reshape(batch_size, n_channels, waveform_length), dim=1
        ) / n_channels
        tmp = stft(shat + delta.repeat_interleave(n_channels, 0))
        phase = torch.nn.functional.normalize(tmp, p=2, dim=-1)
        return Shatmag * phase

class MisiNetwork(torch.nn.Module):
    def __init__(self, n_fft, hop_length, win_length, layer_num=1):
        super(MisiNetwork, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.layer_num = layer_num
    # input: (batch_size, n_channels, freq_bins, spec_time, 2)
    # input: (batch_size, waveform_length)
    # output: (batch_size, n_channels, waveform_length)
    def forward(self, mask, mixture):
        if len(mask.shape) == 4:
            # assumes real mask. zero imaginary part
            mask = torch.stack((mask, torch.zeros_like(mask)), dim=-1)
        batch_size, n_channels, freq_bins, spec_time, _ = mask.shape
        mask = mask.reshape(batch_size * n_channels, freq_bins, spec_time, _)
        waveform_length = mixture.shape[-1]
        def comp_mul(X, Y):
            xre, xim = X.unbind(-1)
            yre, yim = Y.unbind(-1)
            return torch.stack(
                (xre*yre - xim*yim, xre*yim + xim*yre), dim=-1
            )

        stft = lambda x: torch.stft(
            x.reshape(x.shape[:-1].numel(), waveform_length),
            self.n_fft, self.hop_length, self.win_length,
            window=torch.hann_window(self.n_fft)
        )
        istft = lambda X: torchaudio.functional.istft(
            X, self.n_fft, self.hop_length, self.win_length,
            window=torch.hann_window(self.n_fft)
        ).reshape(*X.shape[:-3], waveform_length)

        Shat = comp_mul(mask, stft(mixture).repeat_interleave(n_channels, 0))
        Shatmag = Shat.norm(2, -1, keepdim=True)
        for _ in range(self.layer_num):
            Shat = MisiLayer(self.n_fft, self.hop_length, self.win_length)\
                (Shat, Shatmag, mixture)
        return istft(Shat).reshape(batch_size, n_channels, waveform_length)
