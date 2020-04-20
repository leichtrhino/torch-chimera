
import torch

class ClassicChimera(torch.nn.Module):
    def __init__(self, F, T, C, D, N=400):
        super(ClassicChimera, self).__init__()
        self.freq_bins = F
        self.time = T
        self.n_channels = C
        self.embd_dims = D
        self.n_hidden_states = N
        self.blstm_layer = torch.nn.LSTM(
            F, N, 4, batch_first=True, bidirectional=True
        )
        self.embd_layer = torch.nn.Linear(2*N, F*D)
        self.mask_layer = torch.nn.Linear(D*F, F*C)
    # input: (batch_size, freq_bins, time)
    # output: (batch_size, time*freq_bins, embd_dims)
    # output: (batch_size, n_channels, freq_bins, time)
    def forward(self, x):
        B = x.shape[0]
        out_embd_shape = (B, self.time*self.freq_bins, self.embd_dims)
        out_mask_shape = (B, self.n_channels, self.freq_bins, self.time)
        out_blstm, _ = self.blstm_layer(x.transpose(1, 2)) # (B, T, 2*N)
        out_z = self.embd_layer(out_blstm) # (B, T, F*D)
        out_embd = torch.tanh(out_z) # (B, T, F*D)
        out_embd = out_embd.reshape(*out_embd_shape) # (B, F*T, D)
        out_embd = out_embd \
            / torch.sqrt(torch.sum(out_embd**2, dim=-1, keepdim=True))
        out_mask = self.mask_layer(out_z) # (B, T, F*C)
        out_mask = out_mask.transpose(1, 2).reshape(*out_mask_shape) # (B, C, F, T)
        out_mask = torch.nn.Softmax(dim=1)(out_mask)
        return out_embd, out_mask

class ChimeraPlusPlus(torch.nn.Module):
    def __init__(self, F, T, C, D, N=400):
        super(ChimeraPlusPlus, self).__init__()
        self.freq_bins = F
        self.time = T
        self.n_channels = C
        self.embd_dims = D
        self.n_hidden_states = N
        self.blstm_layer = torch.nn.LSTM(
            F, N, 4, batch_first=True, bidirectional=True
        )
        self.embd_layer = torch.nn.Linear(2*N, F*D)
        self.mask_layer = torch.nn.Linear(2*N, F*C)
    # input: (batch_size, freq_bins, time)
    # output: (batch_size, time*freq_bins, embd_dims)
    # output: (batch_size, n_channels, freq_bins, time)
    def forward(self, x):
        B = x.shape[0]
        out_embd_shape = (B, self.time*self.freq_bins, self.embd_dims)
        out_mask_shape = (B, self.n_channels, self.freq_bins, self.time)
        out_blstm, _ = self.blstm_layer(x.transpose(1, 2)) # (B, T, 2*N)
        out_embd = torch.tanh(self.embd_layer(out_blstm)) # (B, T, F*D)
        out_embd = out_embd.reshape(*out_embd_shape) # (B, F*T, D)
        out_embd = out_embd \
            / torch.sqrt(torch.sum(out_embd**2, dim=-1, keepdim=True))
        out_mask = self.mask_layer(out_blstm) # (B, T, F*C)
        out_mask = out_mask.transpose(1, 2).reshape(*out_mask_shape) # (B, C, F, T)
        out_mask = torch.nn.Softmax(dim=1)(out_mask)
        return out_embd, out_mask
    pass

