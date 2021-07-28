#!/usr/bin/env python
import torch
import torchaudio
import matplotlib.pyplot as plt
from math import pi
from layers import TrainableStftLayer, TrainableIstftLayer

def main():
    window_length, n_fft = 16000, 512
    x = torch.sin(2 * pi * torch.linspace(0, 100, window_length)).unsqueeze(0) +\
    torch.sin(2 * pi * torch.linspace(0, 200, window_length)).unsqueeze(0)
    l1 = TrainableStftLayer(n_fft)
    l2 = TrainableIstftLayer(n_fft)
    X, Xhat = torch.stft(x, n_fft), l1(x.unsqueeze(1))
    y, yhat = torch.istft(X, n_fft), l2(Xhat)
    print(X.shape, Xhat.shape)
    print(y.shape, yhat.shape)

    time = Xhat.shape[-1]
    Xhat = Xhat.reshape(1, 2, n_fft//2+1, time).permute((0, 2, 3, 1))
    X = X.detach().numpy()[0, :, :, 1]
    Xhat = Xhat.detach().numpy()[0, :, :, 1]
    yhat = yhat.detach().numpy()[0]

    '''
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(X)
    ax2.imshow(Xhat)
    plt.show()
    '''

    fig, (ax1, ax2, ax3) = plt.subplots(3)
    ax1.plot(x[0][:1000])
    #ax1.plot(y[0][-1000:])
    ax2.plot(y[0][:1000])
    #ax2.plot(yhat[0][-1000:])
    #yhat = yhat[:, 84:-84]
    ax3.plot(yhat[0][:1000])
    plt.show()

if __name__ =='__main__':
    main()
