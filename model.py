import torch
from torch import nn
import torch.nn.functional as F
from hparams import hparams as hp
from audio import *
from torch.utils.data import DataLoader, Dataset
from utils import num_params

from tqdm import tqdm
import numpy as np

class ResBlock(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.conv1 = nn.Conv2d(dims, dims, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(dims, dims, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(dims)
        self.bn2 = nn.BatchNorm2d(dims)

    def forward(self, x):
        residual = x
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv2(x)
        return x + residual

class ResSkipBlock(nn.Module):
    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.conv0 = nn.Conv2d(in_dims, out_dims, kernel_size=1, bias=False)
        self.conv1 = nn.Conv2d(in_dims, in_dims, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_dims, out_dims, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_dims)
        self.bn2 = nn.BatchNorm2d(in_dims)

    def forward(self, x):
        residual = self.conv0(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv2(x)
        return x + residual
    

class Model(nn.Module):
    def __init__(self, input_dims, output_dims, dilations):
        super().__init__()
        self.conv_in = nn.Conv2d(1, dilations[0], kernel_size=3, padding=1, bias=False)
        self.resnet_layers = nn.ModuleList()
        for i, size in enumerate(dilations[1:]):
            if size == dilations[i]:
                self.resnet_layers.append(ResBlock(size))
            else:
                self.resnet_layers.append(ResSkipBlock(dilations[i], size))
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.fcdims = dilations[-1]*np.product(input_dims)//4
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(self.fcdims, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, output_dims)
        self.sigmoid = nn.Sigmoid()
        num_params(self)
    
    def forward(self, x):
        x = self.conv_in(x)
        for layer in self.resnet_layers:
            x = layer(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.bn1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

    def generate(self, device, path):
        """Given a waveform, generate the vocal-only spectrogram slices.
        Another network will need to convert the spectrogram slices back
        into waveforms that can then be concatenated"""

        self.eval()
        window = hp.hop_size*hp.stft_frames - 1
        stride = hp.hop_size
        wav = load_wav(path)
        count = len(wav)
        i = 0
        output = []
        while (i+window <= count):
            sample = wav[i:i+window]
            x = melspectrogram(sample)
            _x = x[np.newaxis,np.newaxis,:,:]
            _x = torch.FloatTensor(_x)
            _x = _x.to(device)
            _y = self.forward(_x)
            y = _y.to(torch.device('cpu')).detach().numpy()
            if hp.y_tsfm is not None:
                y = y > hp.y_tsfm
            #else:
            #    z = y
            width = x.shape[1]
            z = x[:,width//2]*y
            output.append(z)
            i += stride
        return np.stack(output).astype(np.float32).T[:,0,:]
        

def build_model():
    model = Model((hp.num_mels, 28), hp.num_mels, hp.res_dims)
    return model
