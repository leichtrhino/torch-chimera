
import torch

try:
    import torchchimera
except:
    # attempts to import local module
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    import torchchimera
from torchchimera.models.chimera import ChimeraMagPhasebook
from torchchimera.models.misi import TrainableMisiNetwork

from _training_common import AdaptedChimeraMagPhasebook
from _training_common import AdaptedChimeraMagPhasebookWithMisi

def build_model(model_type, **kwargs):
    if model_type not in ('ChimeraMagPhasebook', 'ChimeraMagPhasebookWithMisi'):
        raise RuntimeError(f'model type "{model_type}" is not available')
    chimera = ChimeraMagPhasebook(
        kwargs['bin_num'],
        kwargs['n_channel'],
        kwargs['embedding_dim'],
        N=kwargs['n_hidden'],
        num_layers=kwargs['n_lstm_layer'],
        residual_base=kwargs['residual']
    )
    if model_type == 'ChimeraMagPhasebook':
        model = AdaptedChimeraMagPhasebook(chimera, kwargs['stft_setting'])
    elif model_type == 'ChimeraMagPhasebookWithMisi':
        misi = TrainableMisiNetwork(
            kwargs['stft_setting'].n_fft, kwargs['n_misi_layer'])
        model = AdaptedChimeraMagPhasebookWithMisi(
            chimera, misi, kwargs['stft_setting'])

    return model

def build_optimizer(model, **kwargs):
    optimizer = torch.optim.Adam(model.parameters(), kwargs['lr'])
    return optimizer

def load_model(checkpoint_path, model_type, **kwargs):
    if model_type is not None and \
       model_type not in ('ChimeraMagPhasebook', 'ChimeraMagPhasebookWithMisi'):
        raise RuntimeError(f'model type "{model_type}" is not available')
    checkpoint = torch.load(checkpoint_path)
    chimera = ChimeraMagPhasebook(
        checkpoint['model']['parameter']['bin_num'],
        checkpoint['model']['parameter']['n_channel'],
        checkpoint['model']['parameter']['embedding_dim'],
        N=checkpoint['model']['parameter']['n_hidden'],
        num_layers=checkpoint['model']['parameter']['n_lstm_layer'],
        residual_base=checkpoint['model']['parameter']['residual_base'],
    )
    if (model_type == None or model_type == 'ChimeraMagPhasebook') and\
       checkpoint['model']['type'] == 'ChimeraMagPhasebook':
        model = AdaptedChimeraMagPhasebook(chimera, kwargs['stft_setting'])
        model.load_state_dict(checkpoint['model']['state_dict'])

    elif model_type == 'ChimeraMagPhasebook' and\
        checkpoint['model']['type'] == 'ChimeraMagPhasebookWithMisi':
        loaded_misi = TrainableMisiNetwork(
            kwargs['stft_setting'].n_fft,
            checkpoint['model']['parameter']['n_misi_layer']
        )
        loaded_model = AdaptedChimeraMagPhasebookWithMisi(
            chimera, loaded_misi, kwargs['stft_setting'])
        loaded_model.load_state_dict(checkpoint['model']['state_dict'])
        model = AdaptedChimeraMagPhasebook(
            loaded_model.chimera, kwargs['stft_setting'])

    elif model_type == 'ChimeraMagPhasebookWithMisi' and\
         checkpoint['model']['type'] == 'ChimeraMagPhasebook':
        misi = TrainableMisiNetwork(
            kwargs['stft_setting'].n_fft, kwargs['n_misi_layer'])
        loaded_model = AdaptedChimeraMagPhasebook(
            chimera, kwargs['stft_setting'])
        loaded_model.load_state_dict(checkpoint['model']['state_dict'])
        model = AdaptedChimeraMagPhasebookWithMisi(
            loaded_model.chimera, misi, kwargs['stft_setting'])

    elif (model_type == None or model_type == 'ChimeraMagPhasebookWithMisi') and\
         checkpoint['model']['type'] == 'ChimeraMagPhasebookWithMisi':
        loaded_misi = TrainableMisiNetwork(
            kwargs['stft_setting'].n_fft,
            checkpoint['model']['parameter']['n_misi_layer']
        )
        loaded_model = AdaptedChimeraMagPhasebookWithMisi(
            chimera, loaded_misi, kwargs['stft_setting'])
        loaded_model.load_state_dict(checkpoint['model']['state_dict'])
        if not kwargs['n_misi_layer']:
            pass
        elif kwargs['n_misi_layer'] < len(loaded_misi.misi_layers):
            loaded_misi.misi_layers =\
                loaded_misi.misi_layers[:kwargs['n_misi_layer']]
        elif kwargs['n_misi_layer'] > len(loaded_misi.misi_layers):
            while kwargs['n_misi_layer'] > len(loaded_misi.misi_layers):
                loaded_misi.add_layer()
        model = AdaptedChimeraMagPhasebookWithMisi(
            chimera, loaded_misi, kwargs['stft_setting'])

    update_args = {
        'model_type': model_type if model_type else checkpoint['model']['type'],
        'bin_num': checkpoint['model']['parameter']['bin_num'],
        'n_channel': checkpoint['model']['parameter']['n_channel'],
        'embedding_dim': checkpoint['model']['parameter']['embedding_dim'],
        'n_hidden': checkpoint['model']['parameter']['n_hidden'],
        'n_lstm_layer': checkpoint['model']['parameter']['n_lstm_layer'],
        'residual': checkpoint['model']['parameter']['residual_base'],
        'n_misi_layer': kwargs.get('n_misi_layer', None) \
        if 'n_misi_layer' not in checkpoint['model']['parameter'] \
        or not checkpoint['model']['parameter']['n_misi_layer'] \
        else checkpoint['model']['parameter']['n_misi_layer'],
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
                'type': 'ChimeraMagPhasebook'\
                if type(model) == AdaptedChimeraMagPhasebook
                else 'ChimeraMagPhasebookWithMisi'
                if type(model) == AdaptedChimeraMagPhasebookWithMisi
                else None,
                'parameter': {
                    'n_hidden': kwargs['n_hidden'],
                    'n_channel': kwargs['n_channel'],
                    'embedding_dim': kwargs['embedding_dim'],
                    'bin_num': kwargs['bin_num'],
                    'residual_base': kwargs['residual'],
                    'n_lstm_layer': kwargs['n_lstm_layer'],
                    'n_misi_layer': kwargs['n_misi_layer']
                    if type(model) == AdaptedChimeraMagPhasebookWithMisi
                    else None,
                },
                'state_dict': model.state_dict(),
            },
            'optimizer': {
                'type': kwargs['loss_function'],
                'epoch': kwargs['epoch'],
                'parameter': {
                    'lr': kwargs['lr'],
                },
                'state_dict': optimizer.state_dict() if optimizer else None,
            },
        },
        checkpoint_path
    )

