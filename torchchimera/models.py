
import torch
from math import pi

# input: (batch_size, freq_bins, time)
# output: (batch_size, time, 2*n_hidden_states)
class ChimeraBase(torch.nn.Module):
    def __init__(self, n_freq_bins, n_hidden_states, dropout=0.3):
        super(ChimeraBase, self).__init__()
        self.blstm_layer = torch.nn.LSTM(
            n_freq_bins, n_hidden_states, 4,
            batch_first=True, bidirectional=True,
            dropout=dropout
        )
    def forward(self, x):
        out_blstm, _ = self.blstm_layer(x.transpose(1, 2)) # (B, T, 2*N)
        return out_blstm

class EmbeddingHead(torch.nn.Module):
    def __init__(self, input_dim, freq_bins, time, embed_dim):
        super(EmbeddingHead, self).__init__()
        self.input_dim = input_dim
        self.freq_bins = freq_bins
        self.time = time
        self.embed_dim = embed_dim
        self.embed_layer = torch.nn.Linear(input_dim, freq_bins*embed_dim)
    def forward(self, x):
        batch_size = x.shape[0]
        out_embed = torch.tanh(self.embed_layer(x)) # (B, T, F*D)
        out_embed = out_embed.reshape(
            batch_size, self.time, self.freq_bins, self.embed_dim # (B,T,F,D)
        ).transpose(1, 2).reshape( # (B, F, T, D)
            batch_size, self.freq_bins*self.time, self.embed_dim # (B, F*T, D)
        )
        out_embed = out_embed /\
            out_embed.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
        return out_embed

class BaseMaskHead(torch.nn.Module):
    def __init__(self, input_dim, freq_bins, time, n_channels,
                 output_dim=1, activation=lambda x: x):
        super(BaseMaskHead, self).__init__()
        self.input_dim = input_dim
        self.freq_bins = freq_bins
        self.time = time
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
            batch_size, self.time,
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
            return torch.matmul(x, self.codebook)
        elif mode == 'argmax':
            return self.codebook[torch.argmax(x, dim=-1)]
        else:
            raise ValueError(f'invalid mode "{mode}"')

class TrainableCodebookMaskHead(torch.nn.Module):
    def __init__(self, codebook_size):
        super(TrainableCodebookMaskHead, self).__init__()
        self.codebook_layer = torch.nn.Linear(codebook_size, 1, bias=False)
    def forward(self, x):
        return self.codebook_layer(x).squeeze(-1)

class ChimeraClassic(torch.nn.Module):
    def __init__(self, F, T, C, D, N=400):
        super(ChimeraClassic, self).__init__()
        self.freq_bins = F
        self.time = T
        self.n_channels = C
        self.embed_dims = D
        self.base = ChimeraBase(F, N)
        self.embed_head = EmbeddingHead(2*N, F, T, D)
        self.mask_head = BaseMaskHead(
            F*D, F, T, C, 1, torch.nn.Softmax(dim=2)
        )
    def forward(self, x):
        batch_size = x.shape[0]
        out_base = self.base(x)
        out_embd = self.embed_head(out_base)
        out_mask = self.mask_head(out_embd.reshape(
            batch_size, self.time, self.freq_bins*self.embed_dims
        ))
        return out_embd, out_mask

class ChimeraPlusPlus(torch.nn.Module):
    def __init__(self, F, T, C, D, N=400):
        super(ChimeraPlusPlus, self).__init__()
        self.base = ChimeraBase(F, N)
        self.embed_head = EmbeddingHead(2*N, F, T, D)
        self.mask_head_base = BaseMaskHead(
            2*N, F, T, C, 3, torch.nn.Softmax(dim=-1)
        )
        self.mask_head = CodebookMaskHead(1.0*torch.arange(3))
    def forward(self, x):
        out_base = self.base(x)
        return self.embed_head(out_base), \
            self.mask_head(self.mask_head_base(out_base))

class ChimeraMagPhasebook(torch.nn.Module):
    def __init__(self, F, T, C, D, N=400):
        super(ChimeraMagPhasebook, self).__init__()
        self.base = ChimeraBase(F, N)
        self.embed_head = EmbeddingHead(2*N, F, T, D)
        self.mag_base = BaseMaskHead(
            2*N, F, T, C, 3, torch.nn.Softmax(dim=-1)
        )
        self.mag_head = CodebookMaskHead(1.0*torch.arange(3))
        self.phase_base = BaseMaskHead(
            2*N, F, T, C, 8, torch.nn.Softmax(dim=-1)
        )
        self.phase_head = CodebookMaskHead(2*pi*torch.arange(-3, 5)/8)
    # valid output modes are: 'mag', 'magp', 'phase', 'phasep', 'com'
    # ['mag', 'phasep', 'com'] for L_{CHI++}(=L_{DC}+L_{MI}), L_{MI}
    # ['com'] for L_{WA}, and waveform inference at test
    def forward(self, x, outputs=['mag', 'phasep', 'com']):
        cossin = lambda x: torch.stack((torch.cos(x), torch.sin(x)), dim=-1)
        out_base = self.base(x)
        out_embed = self.embed_head(out_base)
        out_mag_base = self.mag_base(out_base)
        out_mag = self.mag_head(out_mag_base)
        out_phase_base = self.phase_base(out_base)
        out_phase = self.phase_head(out_phase_base)
        out_com = out_mag.unsqueeze(-1) * cossin(out_phase)
        out_masks = tuple(
            out_mag if mode == 'mag' else
            out_mag_base if mode == 'magp' else
            out_phase if mode == 'phase' else
            out_phase_base if mode == 'phasep' else
            out_com if mode == 'com' else None
            for mode in outputs
        )
        return out_embed, out_masks

class ChimeraCombook(torch.nn.Module):
    def __init__(self, F, T, C, D, N=400):
        super(RefactoredChimeraCombook, self).__init__()
        self.base = ChimeraBase(F, N)
        self.embed_head = EmbeddingHead(2*N, F, T, D)
        self.mask_head = CombookMaskHead(2*N, F, T, C)
    def forward(self, x):
        out_base = self.base(x)
        return self.embed_head(out_base), self.mask_head(out_mask)
