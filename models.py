
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
            / torch.sqrt(torch.sum(out_embd**2, dim=-1, keepdim=True).clamp(min=1e-12))
        out_mask = self.mask_layer(out_z) # (B, T, F*C)
        out_mask = out_mask.transpose(1, 2).reshape(*out_mask_shape) # (B, C, F, T)
        out_mask = torch.nn.Softmax(dim=1)(out_mask)
        return out_embd, out_mask

class ChimeraPlusPlus(torch.nn.Module):
    def __init__(self, F, T, C, D, N=400, activation='softmax'):
        super(ChimeraPlusPlus, self).__init__()
        self.freq_bins = F
        self.time = T
        self.n_channels = C
        self.embd_dims = D
        self.n_hidden_states = N
        self.activation = activation
        self.blstm_layer = torch.nn.LSTM(
            F, N, 4, batch_first=True, bidirectional=True
        )
        self.embd_layer = torch.nn.Linear(2*N, F*D)
        if self.activation in ('softmax', 'doubled_sigmoid', 'clipped_relu'):
            self.mask_layer = torch.nn.Linear(2*N, F*C)
        elif self.activation in ('convex_softmax',):
            self.magbook_size = 3
            self.magbook = torch.arange(self.magbook_size)
            self.mask_layer = torch.nn.Linear(2*N, F*C*self.magbook_size)
            # input: (batch_size, freq_bins, time)
    # output: (batch_size, time*freq_bins, embd_dims)
    # output: (batch_size, n_channels, freq_bins, time)
    def forward(self, x):
        B = x.shape[0]
        out_embd_shape = (B, self.time*self.freq_bins, self.embd_dims)
        out_blstm, _ = self.blstm_layer(x.transpose(1, 2)) # (B, T, 2*N)
        out_embd = torch.tanh(self.embd_layer(out_blstm)) # (B, T, F*D)
        out_embd = out_embd.reshape(*out_embd_shape) # (B, F*T, D)
        out_embd = out_embd \
            / torch.sqrt(torch.sum(out_embd**2, dim=-1, keepdim=True).clamp(min=1e-12))

        if self.activation in ('softmax', 'doubled_sigmoid', 'clipped_relu'):
            out_mask_shape = (B, self.n_channels, self.freq_bins, self.time)
            out_mask = self.mask_layer(out_blstm) # (B, T, F*C)
            out_mask = out_mask.transpose(1, 2) # (B, C*F, T)
            out_mask = out_mask.reshape(*out_mask_shape) # (B, C, F, T)
            if self.activation == 'softmax':
                out_mask = torch.nn.Softmax(dim=1)(out_mask)
            elif self.activation == 'doubled_sigmoid':
                out_mask = 2 * torch.nn.Sigmoid()(out_mask)
            elif self.activation == 'clipped_relu':
                out_mask = torch.clamp(out_mask, min=0., max=2.)
        elif self.activation in ('convex_softmax',):
            out_mask_shape = (B, self.n_channels, self.freq_bins, self.time,
                              self.magbook_size)
            out_mask = self.mask_layer(out_blstm) # (B, T, 3*F*C)
            out_mask = out_mask.reshape(B, self.time, self.n_channels, self.freq_bins, self.magbook_size) # (B, T, C, F, 3)
            out_mask = out_mask.permute(0, 2, 3, 1, 4) # (B, C, F, T, 3)
            out_mask = torch.nn.Softmax(dim=-1)(out_mask)
            out_mask = torch.matmul(out_mask, self.magbook.type_as(out_mask))
        else:
            raise ValueError(f'invalid activation function {self.activation}')

        return out_embd, out_mask

