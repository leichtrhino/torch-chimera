
import torch
from math import pi

# input: (batch_size, freq_bins, time)
# output: (batch_size, time, 2*n_hidden_states)
class ChimeraBase(torch.nn.Module):
    def __init__(self, n_freq_bins, n_hidden_states):
        super(ChimeraBase, self).__init__()
        self.blstm_layer = torch.nn.LSTM(
            n_freq_bins, n_hidden_states, 4,
            batch_first=True, bidirectional=True
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
            batch_size, self.freq_bins*self.time, self.embed_dim) # (B, F*T, D)
        out_embed_mag = torch.sqrt(
            torch.sum(out_embed**2, dim=-1, keepdim=True).clamp(min=1e-12)
        )
        out_embed = out_embed / out_embed_mag
        return out_embed

class _GeneralMaskHead(torch.nn.Module):
    def __init__(
            self, input_dim, freq_bins, time, n_channels, activation,
            output_dim=1, codebook=None, codebook_size=1
    ):
        super(_GeneralMaskHead, self).__init__()
        self.input_dim = input_dim
        self.freq_bins = freq_bins
        self.time = time
        self.n_channels = n_channels
        self.output_dim = output_dim
        self.mask_layer = torch.nn.Linear(
            input_dim, freq_bins*n_channels*codebook_size)
        self.activation = activation
        if codebook is not None:
            self.codebook_layer = lambda x: torch.matmul(x, codebook)
        elif codebook_size > 1:
            self.codebook_layer = torch.nn.Linear(
                codebook_size, output_dim, bias=False)
        else:
            self.codebook_layer = lambda x: x
        self.codebook_size = codebook_size
    def forward(self, x):
        batch_size = x.shape[0]
        out_mask = self.mask_layer(x) # (B, T, F*C*cb)
        out_mask = out_mask.reshape(
            batch_size, self.time, self.freq_bins, self.n_channels,
            self.codebook_size
        )
        out_mask = self.activation(out_mask)
        out_mask = self.codebook_layer(out_mask).squeeze(-1)
        if len(out_mask.shape) == 5:
            return out_mask.permute(0, 3, 2, 1, 4) # (B, C, F, T, O)
        else:
            return out_mask.permute(0, 3, 2, 1) # (B, C, F, T)

class SoftmaxMaskHead(_GeneralMaskHead):
    def __init__(self, input_dim, freq_bins, time, n_channels):
        super(SoftmaxMaskHead, self).__init__(
            input_dim, freq_bins, time, n_channels,
            torch.nn.Softmax(dim=-2) # channel dim now: -2
        )

class DoubledSigmoidMaskHead(_GeneralMaskHead):
    def __init__(self, input_dim, freq_bins, time, n_channels):
        super(DoubledSigmoidMaskHead, self).__init__(
            input_dim, freq_bins, time, n_channels,
            lambda x: 2 * torch.sigmoid(x)
        )

class ClippedReLUMaskHead(_GeneralMaskHead):
    def __init__(self, input_dim, freq_bins, time, n_channels):
        super(ClippedReLUHead, self).__init__(
            input_dim, freq_bins, time, n_channels,
            lambda x: torch.clamp(x, min=0, max=2),
        )

class ConvexSoftmaxMaskHead(_GeneralMaskHead):
    def __init__(self, input_dim, freq_bins, time, n_channels):
        super(ConvexSoftmaxMaskHead, self).__init__(
            input_dim, freq_bins, time, n_channels,
            torch.nn.Softmax(dim=-1),
            codebook=1.0*torch.arange(3),
            codebook_size=3,
        )

class MagbookMaskHead(_GeneralMaskHead):
    def __init__(self, input_dim, freq_bins, time, n_channels):
        super(MagbookMaskHead, self).__init__(
            input_dim, freq_bins, time, n_channels,
            torch.nn.Softmax(dim=-1),
            codebook=1.0*torch.arange(3),
            codebook_size=3,
        )

class PhasebookMaskHead(_GeneralMaskHead):
    def __init__(self, input_dim, freq_bins, time, n_channels):
        super(PhasebookMaskHead, self).__init__(
            input_dim, freq_bins, time, n_channels,
            torch.nn.Softmax(dim=-1),
            codebook=2*pi*torch.arange(-3, 5),
            codebook_size=8,
        )

class CombookMaskHead(_GeneralMaskHead):
    def __init__(self, input_dim, freq_bins, time, n_channels):
        super(CombookMaskHead, self).__init__(
            input_dim, freq_bins, time, n_channels,
            torch.nn.Softmax(dim=-1),
            output_dim=2,
            codebook_size=12,
        )

class ChimeraClassic(torch.nn.Module):
    def __init__(self, F, T, C, D, N=400):
        super(ChimeraClassic, self).__init__()
        self.freq_bins = F
        self.time = T
        self.n_channels = C
        self.embed_dims = D
        self.base = ChimeraBase(F, N)
        self.embed_head = EmbeddingHead(2*N, F, T, D)
        self.mask_head = SoftmaxMaskHead(F*D, F, T, C)
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
        self.mask_head = ConvexSoftmaxMaskHead(2*N, F, T, C)
    def forward(self, x):
        out_base = self.base(x)
        return self.embed_head(out_base), self.mask_head(out_base)

class ChimeraMagbookPhasebook(torch.nn.Module):
    def __init__(self, F, T, C, D, N=400):
        super(ChimeraMagbookPhasebook, self).__init__()
        self.base = ChimeraBase(F, N)
        self.embed_head = EmbeddingHead(2*N, F, T, D)
        self.mag_mask_head = MagbookMaskHead(2*N, F, T, C)
        self.phase_mask_head = PhasebookMaskHead(2*N, F, T, C)
    def forward(self, x):
        out_base = self.base(x)
        out_embed = self.embed_head(out_base)
        out_mag_mask = self.mag_mask_head(out_base)
        out_phase_mask = self.phase_mask_head(out_base)
        return out_embed, out_mag_mask, out_phase_mask

class ChimeraCombook(torch.nn.Module):
    def __init__(self, F, T, C, D, N=400):
        super(RefactoredChimeraCombook, self).__init__()
        self.base = ChimeraBase(F, N)
        self.embed_head = EmbeddingHead(2*N, F, T, D)
        self.mask_head = CombookMaskHead(2*N, F, T, C)
    def forward(self, x):
        out_base = self.base(x)
        return self.embed_head(out_base), self.mask_head(out_mask)

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

