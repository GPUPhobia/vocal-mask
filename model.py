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
        self.conv0 = nn.Conv2d(in_dims, out_dims, kernel_size=1, stride=1, bias=False)
        self.conv1 = nn.Conv2d(in_dims, out_dims, kernel_size=3, padding=1, stride=1, bias=False)
        self.conv2 = nn.Conv2d(out_dims, out_dims, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_dims)
        self.bn2 = nn.BatchNorm2d(out_dims)

    def forward(self, x):
        residual = self.conv0(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv2(x)
        return x + residual

class Conv3x3(nn.Module):
    def __init__(self, in_planes, out_planes, activation):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_planes)
        self.activation = activation

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))

class FC(nn.Module):
    def __init__(self, indims, outdims, activation):
        super().__init__()
        self.fc = nn.Linear(indims, outdims)
        self.bn = nn.BatchNorm1d(outdims)
        self.activation = activation

    def forward(self, x):
        return self.activation(self.bn(self.fc(x)))

class ConvNet(nn.Module):
    def __init__(self, input_dims, output_dims):
        super().__init__()
        self.activation = F.relu
        #self.activation = F.leaky_relu
        self.conv1 = Conv3x3(1, 32, self.activation)
        self.conv2 = Conv3x3(32, 16, self.activation)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout(p=0.25)
        self.conv3 = Conv3x3(16, 64, self.activation)
        self.conv4 = Conv3x3(64, 16, self.activation)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.dropout2 = nn.Dropout(p=0.5)
        fcdims = 16*np.product([dim//4 for dim in input_dims])
        self.fc1 = FC(fcdims, 128, self.activation)
        self.dropout3 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, output_dims)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout3(x)
        x = self.fc2(x)
        return x

class ResNet(nn.Module):
    def __init__(self, input_dims, output_dims, res_dims):
        super().__init__()
        in_filters = res_dims[0][0]
        out_filters = res_dims[-1][1]
        self.conv_in = nn.Conv2d(1, in_filters, kernel_size=3, padding=1, bias=False)
        self.resnet_layers = nn.ModuleList()
        downsample_factor = 1
        for res_dim in res_dims:
            in_planes, out_planes, downsample = res_dim
            if in_planes == out_planes:
                self.resnet_layers.append(ResBlock(in_planes))
            else:
                self.resnet_layers.append(ResSkipBlock(in_planes, out_planes))
            if downsample is not None:
                self.resnet_layers.append(nn.MaxPool2d(kernel_size=downsample[0], stride=downsample[1]))
                downsample_factor *= downsample[1]
        self.bn1 = nn.BatchNorm2d(out_filters)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fcdims = out_filters*np.prod([dim//downsample_factor for dim in input_dims])
        self.fc1 = FC(self.fcdims, 128, F.relu)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, output_dims)

    def forward(self, x):
        x = self.conv_in(x)
        for layer in self.resnet_layers:
            x = layer(x)
        x = F.relu(self.bn1(x))
        #x = self.maxpool1(x)
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x
    

class Model(nn.Module):
    def __init__(self, input_dims, output_dims, model_type):
        super().__init__()
        if model_type=='convnet':
            self.cnn = ConvNet(input_dims, output_dims)
        elif model_type=='resnet':
            self.cnn = ResNet(input_dims, output_dims, hp.res_dims)
        self.sigmoid = nn.Sigmoid()
        num_params(self)
    
    def forward(self, x):
        x = self.cnn(x)
        x = self.sigmoid(x)
        return x

    def generate_eval(self, device, wav):
        """Given a waveform, generate the vocal-only spectrogram slices.
        Another network will need to convert the spectrogram slices back
        into waveforms that can then be concatenated"""

        self.eval()
        window = hp.hop_size*hp.stft_frames - 1
        stride = hp.hop_size
        count = len(wav)
        i = 0
        output = []
        mask = []
        while (i+window <= count):
            sample = wav[i:i+window]
            x = spectrogram(sample)[0]
            _x = x[np.newaxis,np.newaxis,:,:]
            _x = torch.FloatTensor(_x).to(device)
            _y = self.forward(_x)
            y = _y.to(torch.device('cpu')).detach().numpy()
            if hp.mask_at_eval:
                y = y > 0.5
            z = x[:,hp.stft_frames//2]*y
            output.append(z)
            mask.append(y)
            i += stride
        return (np.vstack(output).astype(np.float32).T, 
                    np.vstack(mask).astype(np.float32).T)

    def generate(self, device, wav):
        self.eval()
        window = hp.hop_size*hp.stft_frames - 1
        stride = hp.hop_size
        count = len(wav)
        output = []
        end = count - (count%stride) - window
        for i in tqdm(range(0, end//stride)):
            x, stftx = spectrogram(wav[i*stride:i*stride+window])
            _x = torch.FloatTensor(x[np.newaxis,np.newaxis,:,:]).to(device)
            _y = self.forward(_x)
            y = _y.to(torch.device('cpu')).detach().numpy()
            if hp.mask_at_eval:
                y = y > hp.eval_mask_threshold
            z = stftx[:,hp.stft_frames//2]*y
            if not hp.mask_at_eval:
                z = z*(z > hp.noise_gate)
            output.append(z)
        S = np.vstack(output).T
        return inv_spectrogram(S)

def build_model():
    fft_bins = hp.fft_size//2+1
    model = Model((fft_bins, hp.stft_frames), fft_bins, hp.model_type)
    return model
