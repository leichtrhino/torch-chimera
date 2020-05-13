
import torch

class MixTransform(torch.nn.Module):
    def __init__(self, source_lists=[(0, 1, 2), 3], source_coeffs=None):
        super(MixTransform, self).__init__()
        self.source_lists = [
            torch.Tensor(s if hasattr(s, '__len__') else [s]).long()
            for s in source_lists
        ]
        if source_coeffs is None:
            self.source_coeffs = [
                torch.ones(len(s)) if hasattr(s, '__len__') else
                torch.ones(1) for s in source_lists
            ]
        else:
            self.source_coeffs = source_coeffs

    def forward(self, sample):
        # weighted sum along dim=-2 (dim=-1 are waveforms)
        return torch.stack([
            torch.sum(
                sc.unsqueeze(-1) * sample.index_select(dim=-2, index=sl),
                dim=-2
            )
            for sl, sc in zip(self.source_lists, self.source_coeffs)
        ], dim=-2)

