
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

class Stft(torch.nn.Module):
    def __init__(self, n_fft, hop_length=None, win_length=None, window=None):
        super(Stft, self).__init__()
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
    def forward(self, x):
        waveform_length = x.shape[-1]
        y = torch.stft(
            x.reshape(x.shape[:-1].numel(), waveform_length),
            self.n_fft,
            self.hop_length,
            self.win_length,
            window=self.window
        )
        _, freq, time, _ = y.shape
        return y.reshape(*x.shape[:-1], freq, time, 2)

class Istft(torch.nn.Module):
    def __init__(self, n_fft, hop_length=None, win_length=None, window=None):
        super(Istft, self).__init__()
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
    def forward(self, x):
        freq, time = x.shape[-3], x.shape[-2]
        y = torchaudio.functional.istft(
            x.reshape(x.shape[:-3].numel(), freq, time, 2),
            self.n_fft,
            self.hop_length,
            self.win_length,
            window=self.window
        )
        waveform_length = y.shape[-1]
        return y.reshape(*x.shape[:-3], waveform_length)

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

def comp_mul(X, Y):
    (X_re, X_im), (Y_re, Y_im) = X.unbind(-1), Y.unbind(-1)
    return torch.stack((
        X_re * Y_re - X_im * Y_im,
        X_re * Y_im + X_im * Y_re
    ), dim=-1)

def make_x_in(batch, args):
    window = torch.sqrt(torch.hann_window(args.n_fft, device=batch.device))
    stft = Stft(args.n_fft, args.hop_length, args.win_length, window)
    s, x = batch, batch.sum(dim=1)
    S, X = stft(s), stft(x)
    return s, x, S, X

def forward(model, x_in, args):
    s, x, S, X = x_in
    embd, (mag, com), _ = model(
        torch.log10(X.norm(p=2, dim=-1).clamp(min=1e-40)),
        outputs=['mag', 'com']
    )
    return embd, (mag, com)

def compute_loss(x_in, y_pred, args):
    window = torch.sqrt(torch.hann_window(args.n_fft, device=x_in[0].device))
    istft = Istft(args.n_fft, args.hop_length, args.win_length, window)
    s, x, S, X = x_in
    embd, (mag, com) = y_pred
    if args.loss_function == 'chimera++':
        Y = dc_label_matrix(S)
        weight = dc_weight_matrix(X)
        alpha = 0.975
        loss_dc = alpha * loss_dc_deep_lda(embd, Y, weight)
        if args.permutation_free:
            loss_mi = (1-alpha) * permutation_free(loss_mi_tpsa)(mag, X, S, gamma=2.)
        else:
            loss_mi = (1-alpha) * loss_mi_tpsa(mag, X, S, gamma=2.)
        loss = loss_dc + loss_mi
    elif args.loss_function == 'wave-approximation':
        Shat = comp_mul(com, X.unsqueeze(1))
        shat = istft(Shat)
        waveform_length = min(s.shape[-1], shat.shape[-1])
        s = s[:, :, :waveform_length]
        shat = shat[:, :, :waveform_length]
        if args.permutation_free:
            loss = permutation_free(loss_wa)(shat, s)
        else:
            loss = loss_wa(shat, s)
    return loss

def predict_waveform(model, mixture, args):
    window = torch.sqrt(torch.hann_window(args.n_fft)).to(args.device)
    stft = Stft(args.n_fft, args.hop_length, args.win_length, window)
    istft = Istft(args.n_fft, args.hop_length, args.win_length, window)
    X = stft(mixture)
    _, (com,), _ = model(
        torch.log10(X.norm(p=2, dim=-1).clamp(min=1e-40)),
        outputs=['com']
    )
    Shat = comp_mul(com, X.unsqueeze(1))
    return istft(Shat)

