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
    def __init__(self, in_dims, out_dims):
        super().__init__()
        if in_dims == out_dims:
            stride = 1
            self.downsample = None
        else:
            stride = 2
            self.downsample = nn.Sequential(
                    nn.Conv2d(in_dims, out_dims, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_dims)
            )
        self.conv1 = nn.Conv2d(in_dims, out_dims, kernel_size=3, padding=1, stride=stride, bias=False)
        self.conv2 = nn.Conv2d(out_dims, out_dims, kernel_size=3, padding=1, stride=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(out_dims)
        self.bn2 = nn.BatchNorm2d(out_dims)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class PreActResBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        if in_planes == out_planes:
            stride = 1
            self.shortcut = None
        else:
            stride = 2
            self.shortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        identity = x
        if self.shortcut is not None:
            identity = self.shortcut(x)
        out = self.relu(self.bn1(x))
        out = self.conv1(out)
        out = self.relu(self.bn2(out))
        out = self.conv2(out)
        return out + identity

class Conv3x3(nn.Module):
    def __init__(self, in_planes, out_planes, activation, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, groups=groups, bias=False)
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
        block = PreActResBlock
        self.conv_in = nn.Conv2d(1, in_filters, kernel_size=(7,3), 
                                 padding=(3,1), stride=(2,1), bias=False)
        self.resnet_layers = self._build_layers(res_dims, block)
        self.bn = nn.BatchNorm2d(out_filters)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(out_filters, output_dims)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _build_layers(self, res_dims, block):
        layers = [block(*dim) for dim in res_dims]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.resnet_layers(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ResNet18(nn.Module):
    def __init__(self, input_dims, output_dims):
        super().__init__()
        res_dims = [
            (64, 64), (64, 64), (64, 128),
            (128, 128), (128, 128), (128, 256),
            (256, 256), (256, 256), (256, 512),
            (512, 512), (512, 512)
        ]
        self.resnet = ResNet(input_dims, output_dims, res_dims)

    def forward(self, x):
        return self.resnet(x)

class ResNet34(nn.Module):
    def __init__(self, input_dims, output_dims):
        super().__init__()
        res_dims = [
            (64, 64), (64, 64), (64, 64), (64, 128),
            (128, 128), (128, 128), (128, 128), (128, 128), (128, 256),
            (256, 256), (256, 256), (256, 256), (256, 256), (256, 256),
                (256, 256), (256, 512),
            (512, 512), (512, 512), (512, 512)
        ]
        self.resnet = ResNet(input_dims, output_dims, res_dims)

    def forward(self, x):
        return self.resnet(x)

class Model(nn.Module):
    def __init__(self, input_dims, output_dims, model_type):
        super().__init__()
        if model_type=='convnet':
            self.cnn = ConvNet(input_dims, output_dims)
        elif model_type=='resnet18':
            self.cnn = ResNet18(input_dims, output_dims)
        elif model_type=='resnet34':
            self.cnn = ResNet34(input_dims, output_dims)
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
            x = spectrogram(sample, power=hp.mix_power_factor)[0]
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

    def generate(self, device, wav, targets=['vocals','accompaniment']):
        """
        TODO: Zero-pad the spectrogram or waveform so that output is same
        length as input
        """
        self.eval()
        mel_spec, stft = spectrogram(wav, power=hp.mix_power_factor)
        padding = hp.stft_frames//2
        mel_spec = np.pad(mel_spec, ((0,0),(padding,padding)), 'constant', constant_values=0) 
        window = hp.stft_frames
        size = mel_spec.shape[1]
        output_vox = []
        output_bg = []
        end = size - window
        for i in tqdm(range(0, end+1)):
            x = mel_spec[:,i:i+window]
            _x = torch.FloatTensor(x[np.newaxis,np.newaxis,:,:]).to(device)
            _y = self.forward(_x)
            y = _y.to(torch.device('cpu')).detach().numpy()
            if hp.mask_at_eval:
                y = y > hp.eval_mask_threshold
                yb = y <= hp.eval_mask_threshold
            else:
                y = y*(y > hp.noise_gate)
                yb = y <= hp.noise_gate
            z = stft[:,i]*y
            zb = stft[:,i]*yb
            output_vox.append(z)
            output_bg.append(zb)
        S_vox = np.vstack(output_vox).T
        S_bg = np.vstack(output_bg).T
        estimates = {}
        if 'vocals' in targets:
            estimates['vocals'] = inv_spectrogram(S_vox)
        if 'accompaniment' in targets:
            estimates['accompaniment'] = inv_spectrogram(S_bg)
        return estimates


def build_model():
    fft_bins = hp.fft_size//2+1
    model = Model((fft_bins, hp.stft_frames), fft_bins, hp.model_type)
    return model
