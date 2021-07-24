
import torch
from math import pi, cos, sin

def _initialize_lstm_weights(lstm_layer, hidden_size):
    N = hidden_size
    for name, param in lstm_layer.named_parameters():
        N = hidden_size
        if 'weight_ih' in name:
            for i, gain in enumerate([
                    torch.nn.init.calculate_gain('sigmoid'),
                    torch.nn.init.calculate_gain('sigmoid'),
                    torch.nn.init.calculate_gain('tanh'),
                    torch.nn.init.calculate_gain('sigmoid')
            ]):
                torch.nn.init.xavier_uniform_(param[i*N:(i+1)*N], gain)
        elif 'weight_hh' in name:
            for i, gain in enumerate([
                    torch.nn.init.calculate_gain('sigmoid'),
                    torch.nn.init.calculate_gain('sigmoid'),
                    torch.nn.init.calculate_gain('tanh'),
                    torch.nn.init.calculate_gain('sigmoid')
            ]):
                torch.nn.init.orthogonal_(param[i*N:(i+1)*N], gain)
        elif 'bias' in name:
            param.data.fill_(0)
            param.data[N:2*N].fill_(1) # for forget gate

# input: (batch_size, freq_bins, time)
# output: (batch_size, time, 2*n_hidden_states)
class ChimeraBase(torch.nn.Module):
    def __init__(self, n_freq_bins, n_hidden_states, num_layers=4, dropout=0.3):
        super(ChimeraBase, self).__init__()
        self.hidden_size = n_hidden_states
        self.num_layers = num_layers
        self.blstm_layer = torch.nn.LSTM(
            n_freq_bins, n_hidden_states, self.num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout
        )
        _initialize_lstm_weights(self.blstm_layer, self.hidden_size)

    def forward(self, x, initial_states=None):
        if initial_states is None:
            return self.blstm_layer(x.transpose(1, 2)) # (B, T, 2*N)
        else:
            return self.blstm_layer(x.transpose(1, 2), initial_states) # (B, T, 2*N)

class ResidualChimeraBase(torch.nn.Module):
    def __init__(self, n_freq_bins, n_hidden_states, num_layers=4, dropout=0.3):
        super(ResidualChimeraBase, self).__init__()
        self.hidden_size = n_hidden_states
        self.num_layers = num_layers
        self.blstm_layers = torch.nn.ModuleList([
            torch.nn.LSTM(
                n_freq_bins if i == 1 else n_hidden_states*2,
                n_hidden_states,
                batch_first=True,
                bidirectional=True
            )
            for i in range(1, self.num_layers+1)
        ])
        for layer in self.blstm_layers:
            _initialize_lstm_weights(layer, n_hidden_states)
        self.dropout_layer = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(n_freq_bins, n_hidden_states*2)\
            if n_freq_bins != n_hidden_states*2 else torch.nn.Identity()

    def forward(self, x, initial_states=None):
        h_n_list, c_n_list = [], []
        x = x.transpose(1, 2) # B, T, F
        for i in range(self.num_layers):
            if initial_states is None:
                out_lstm, (h_n, c_n) = self.blstm_layers[i](x)
            else:
                h_0, c_0 = initial_states
                out_lstm, (h_n, c_n) = self.blstm_layers[i](
                    x, (h_0[i*2:(i+1)*2], c_0[i*2:(i+1)*2])
                )
            if i < self.num_layers - 1:
                out_lstm = self.dropout_layer(out_lstm)
                x = (self.linear(x) if i == 0 else x) + out_lstm
            else:
                x = out_lstm
            h_n_list.append(h_n)
            c_n_list.append(c_n)
        return x, (torch.cat(h_n_list), torch.cat(c_n_list))

class EmbeddingHead(torch.nn.Module):
    def __init__(self, input_dim, freq_bins, embed_dim,
                 activation=torch.nn.Sigmoid()):
        super(EmbeddingHead, self).__init__()
        self.input_dim = input_dim
        self.freq_bins = freq_bins
        self.embed_dim = embed_dim
        self.embed_layer = torch.nn.Linear(input_dim, freq_bins*embed_dim)
        self.activation = activation
    def forward(self, x):
        batch_size = x.shape[0]
        out_embed = self.activation(self.embed_layer(x)) # (B, T, F*D)
        out_embed = out_embed.reshape(
            batch_size,
            out_embed.shape.numel() // (batch_size * self.embed_dim),
            self.embed_dim # (B,T*F,D)
        )
        out_embed = out_embed /\
            out_embed.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
        return out_embed

class BaseMaskHead(torch.nn.Module):
    def __init__(self, input_dim, freq_bins, n_channels,
                 output_dim=1, activation=lambda x: x):
        super(BaseMaskHead, self).__init__()
        self.input_dim = input_dim
        self.freq_bins = freq_bins
        self.n_channels = n_channels
        self.output_dim = output_dim
        self.activation = activation
        self.mask_layer = torch.nn.Linear(
            input_dim, freq_bins*n_channels*output_dim
        )
    def forward(self, x):
        batch_size = x.shape[0]
        out_mask = self.mask_layer(x) # (B, T, F*C*O)
        out_mask = out_mask.reshape( # (B, T, F, C, O)
            batch_size,
            out_mask.shape.numel() // (
                batch_size *
                self.freq_bins * self.n_channels * self.output_dim
            ),
            self.freq_bins, self.n_channels, self.output_dim
        )
        out_mask = out_mask.permute(0, 3, 2, 1, 4) # (B, C, F, T, O)
        out_mask = self.activation(out_mask)
        return out_mask.squeeze(-1)

class CodebookMaskHead(torch.nn.Module):
    def __init__(self, codebook):
        super(CodebookMaskHead, self).__init__()
        self.codebook = codebook
    def forward(self, x, mode='interpolate'):
        # valid modes are: 'interpolate', 'argmax'
        if mode == 'interpolate':
            return torch.matmul(x, self.codebook.to(x.device))
        elif mode == 'argmax':
            return self.codebook[
                torch.argmax(x.to(codebook.device), dim=-1)
            ].to(x.device)
        else:
            raise ValueError(f'invalid mode "{mode}"')

class TrainableCodebookMaskHead(torch.nn.Module):
    def __init__(self, codebook_size):
        super(TrainableCodebookMaskHead, self).__init__()
        self.codebook_layer = torch.nn.Linear(codebook_size, 1, bias=False)
    def forward(self, x):
        return self.codebook_layer(x).squeeze(-1)

class ChimeraClassic(torch.nn.Module):
    def __init__(self, F, T, C, D, N=400, num_layers=4):
        super(ChimeraClassic, self).__init__()
        self.freq_bins = F
        self.time = T
        self.n_channels = C
        self.embed_dims = D
        self.base = ChimeraBase(F, N, num_layers=num_layers)
        self.embed_head = EmbeddingHead(2*N, F, D)
        self.mask_head = BaseMaskHead(
            F*D, F, C, 1, torch.nn.Softmax(dim=2)
        )
    def forward(self, x, states=None):
        batch_size = x.shape[0]
        out_base, out_states = self.base(x, states)
        out_embd = self.embed_head(out_base)
        out_mask = self.mask_head(out_embd.reshape(
            batch_size, self.time, self.freq_bins*self.embed_dims
        ))
        return out_embd, out_mask, out_states

class ChimeraPlusPlus(torch.nn.Module):
    def __init__(self, F, C, D, N=400, num_layers=4):
        super(ChimeraPlusPlus, self).__init__()
        self.base = ChimeraBase(F, N, num_layers=num_layers)
        self.embed_head = EmbeddingHead(2*N, F, D)
        self.mask_head_base = BaseMaskHead(
            2*N, F, C, 3, torch.nn.Softmax(dim=-1)
        )
        self.mask_head = CodebookMaskHead(1.0*torch.arange(3))
    def forward(self, x, states=None):
        out_base, out_states = self.base(x, states)
        return self.embed_head(out_base), \
            self.mask_head(self.mask_head_base(out_base)), \
            out_states

class ChimeraMagPhasebook(torch.nn.Module):
    def __init__(self, F, C, D, N=400, num_layers=4,
                 embed_activation=torch.nn.Sigmoid(),
                 residual_base=False):
        super(ChimeraMagPhasebook, self).__init__()
        if residual_base:
            self.base = ResidualChimeraBase(F, N, num_layers=num_layers)
        else:
            self.base = ChimeraBase(F, N, num_layers=num_layers)
        self.embed_head = EmbeddingHead(2*N, F, D, embed_activation)
        self.mag_base = BaseMaskHead(
            2*N, F, C, 3, torch.nn.Softmax(dim=-1)
        )
        self.mag_head = CodebookMaskHead(1.0*torch.arange(3))
        self.phase_base = BaseMaskHead(
            2*N, F, C, 8, torch.nn.Softmax(dim=-1)
        )
        self.phase_head = CodebookMaskHead(
            torch.Tensor([
                (cos(theta), sin(theta))
                for theta in 2*pi*torch.arange(-3, 5)/8
            ])
        )

    # valid output modes are: 'mag', 'magp', 'phase', 'phasep', 'com'
    # ['mag', 'phasep', 'com'] for L_{CHI++}(=L_{DC}+L_{MI}), L_{MI}
    # ['com'] for L_{WA}, and waveform inference at test
    def forward(self, x, states=None, outputs=['mag', 'phasep', 'com']):
        cossin = lambda x: torch.stack((torch.cos(x), torch.sin(x)), dim=-1)
        out_base, out_states = self.base(x, states)
        out_embed = self.embed_head(out_base)
        out_mag_base = self.mag_base(out_base)
        out_mag = self.mag_head(out_mag_base)
        out_phase_base = self.phase_base(out_base)
        out_phase_com = self.phase_head(out_phase_base)
        out_phase = torch.atan2(
            *out_phase_com.split(1, dim=-1)[::-1]
        ).squeeze(-1)
        out_com = out_mag.unsqueeze(-1) * cossin(out_phase)
        out_masks = tuple(
            out_mag if mode == 'mag' else
            out_mag_base if mode == 'magp' else
            out_phase if mode == 'phase' else
            out_phase_base if mode == 'phasep' else
            out_com if mode == 'com' else None
            for mode in outputs
        )
        return out_embed, out_masks, out_states

class ChimeraCombook(torch.nn.Module):
    def __init__(self, F, C, D, N=400, num_layers=4,
                 embed_activation=torch.nn.Sigmoid(),
                 residual_base=False):
        super(ChimeraCombook, self).__init__()
        if residual_base:
            self.base = ResidualChimeraBase(F, N, num_layers=num_layers)
        else:
            self.base = ChimeraBase(F, N, num_layers=num_layers)
        self.embed_head = EmbeddingHead(2*N, F, D, embed_activation)
        self.mask_base = BaseMaskHead(
            2*N, F, C, 12, torch.nn.Softmax(dim=-1)
        )
        self.re_mask_head = TrainableCodebookMaskHead(12)
        self.im_mask_head = TrainableCodebookMaskHead(12)
    def forward(self, x, states=None):
        out_base, out_states = self.base(x, states)
        out_mask_base = self.mask_base(out_base)
        return\
            self.embed_head(out_base),\
            torch.stack(
                (
                    self.re_mask_head(out_mask_base),
                    self.im_mask_head(out_mask_base)
                ),
                dim=-1),\
            out_states
