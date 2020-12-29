
import torch

try:
    import torchchimera
except:
    # attempts to import local module
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    import torchchimera
from torchchimera.models.chimera import ChimeraMagPhasebook

from _training_common import AdaptedChimeraMagPhasebook

def build_model(model_type, **kwargs):
    chimera = ChimeraMagPhasebook(
        kwargs['bin_num'],
        kwargs['n_channel'],
        kwargs['embedding_dim'],
        N=kwargs['n_hidden'],
        residual_base=kwargs['residual']
    )
    model = AdaptedChimeraMagPhasebook(chimera, kwargs['stft_setting'])
    return model

def build_optimizer(model, **kwargs):
    optimizer = torch.optim.Adam(model.parameters(), kwargs['lr'])
    return optimizer

def load_model(checkpoint_path, model_type, **kwargs):
    checkpoint = torch.load(checkpoint_path)
    chimera = ChimeraMagPhasebook(
        checkpoint['model']['parameter']['bin_num'],
        checkpoint['model']['parameter']['n_channel'],
        checkpoint['model']['parameter']['embedding_dim'],
        N=checkpoint['model']['parameter']['n_hidden'],
        residual_base=checkpoint['model']['parameter']['residual_base'],
    )
    model = AdaptedChimeraMagPhasebook(chimera, kwargs['stft_setting'])
    model.load_state_dict(checkpoint['model']['state_dict'])

    update_args = {
        'bin_num': checkpoint['model']['parameter']['bin_num'],
        'n_channel': checkpoint['model']['parameter']['n_channel'],
        'embedding_dim': checkpoint['model']['parameter']['embedding_dim'],
        'n_hidden': checkpoint['model']['parameter']['n_hidden'],
        'residual': checkpoint['model']['parameter']['residual_base'],
    }
    return model, update_args

def load_optimizer(checkpoint_path, model, **kwargs):
    checkpoint = torch.load(checkpoint_path)
    optimizer = torch.optim.Adam(
        model.parameters(),
        checkpoint['optimizer']['parameter']['lr']
    )
    optimizer.load_state_dict(checkpoint['optimizer']['state_dict'])

    initial_epoch = checkpoint['optimizer']['epoch']
    update_args = {
        'loss_function': checkpoint['optimizer']['type'],
        'lr': checkpoint['optimizer']['parameter']['lr']
    }
    return optimizer, initial_epoch, update_args

def save_checkpoint(model, optimizer, checkpoint_path, **kwargs):
    torch.save(
        {
            'model': {
                'type': 'ChimeraMagPhasebook',
                'parameter': {
                    'n_hidden': kwargs['n_hidden'],
                    'n_channel': kwargs['n_channel'],
                    'embedding_dim': kwargs['embedding_dim'],
                    'bin_num': kwargs['bin_num'],
                    'residual_base': kwargs['residual'],
                },
                'state_dict': model.state_dict(),
            },
            'optimizer': {
                'type': kwargs['loss_function'],
                'epoch': kwargs['epoch'],
                'parameter': {
                    'lr': kwargs['lr'],
                },
                'state_dict': optimizer.state_dict(),
            },
        },
        checkpoint_path
    )

