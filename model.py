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

def calc_padding(kernel):
    return list((i//2 for i in kernel))

class PreActResBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=(3,3)):
        super().__init__()
        if in_planes == out_planes:
            stride = 1
            self.shortcut = None
        else:
            stride = 2
            self.shortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
        padding = calc_padding(kernel_size)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, stride=1, padding=padding, bias=False)

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
        block = PreActResBlock
        in_filters = res_dims[0][0]
        out_filters = res_dims[-1][1]
        init_padding = calc_padding(hp.init_conv_kernel)
        self.conv_in = nn.Conv2d(1, 
                                 in_filters, 
                                 kernel_size=hp.init_conv_kernel, 
                                 padding=init_padding, 
                                 stride=(2,1), 
                                 bias=False)
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
        layers = [block(*dim, hp.kernel) for dim in res_dims]
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

    def generate_specs(self, device, wav):
        """Generate the vocal-accompaniment separated spectrograms"""

        mask, mel, stft = self.predict_mask(device, wav)
        Mvox, Macc = self.process_mask(mask)
        results = {
            "mask": { "vocals": Mvox, "accompaniment": Macc },
            "stft": stft,
            "spec": mel
        }
        return results

    def generate_wav(self, device, wav):
        """Generate the vocal-accompaniment separated waveforms"""

        S = self.generate_specs(device, wav)
        Svox = self.apply_mask(S["mask"]["vocals"], S["stft"])
        Sacc = self.apply_mask(S["mask"]["accompaniment"], S["stft"])
        estimates = {
            "vocals": inv_spectrogram(Svox),
            "accompaniment": inv_spectrogram(Sacc)
        }
        return estimates

    def predict_mask(self, device, wav):
        """Perform a forward pass through the model to generate the
        vocal soft masks. 
        """

        self.eval()
        _mel_spec, stft = spectrogram(wav, power=hp.mix_power_factor)
        padding = hp.stft_frames//2
        mel_spec = np.pad(_mel_spec, ((0,0),(padding,padding)), 'constant', constant_values=0) 
        window = hp.stft_frames
        size = mel_spec.shape[1]
        mask = []
        end = size - window
        for i in tqdm(range(0, end+1)):
            x = mel_spec[:,i:i+window]
            _x = torch.FloatTensor(x[np.newaxis,np.newaxis,:,:]).to(device)
            _y = self.forward(_x)
            y = _y.to(torch.device('cpu')).detach().numpy()
            mask.append(y)
        mask = np.vstack(mask).T
        return mask, _mel_spec, stft

    def apply_mask(self, mask, stft):
        return stft*mask

    def process_mask(self, mask):
        if hp.mask_at_eval:
            Mvox, Macc = self.get_binary_mask(mask)
        else:
            Mvox, Macc = self.get_soft_mask(mask)
        return Mvox, Macc

    def get_binary_mask(self, mask):
        Mvox = mask > hp.eval_mask_threshold
        Macc = mask <= hp.eval_mask_threshold
        return Mvox, Macc

    def get_soft_mask(self, mask):
        Mvox = mask * (mask > hp.eval_mask_threshold)
        inv_mask = 1 - mask
        Macc = inv_mask * (inv_mask > hp.eval_mask_threshold)
        return Mvox, Macc
        


def build_model():
    fft_bins = hp.fft_size//2+1
    model = Model((fft_bins, hp.stft_frames), fft_bins, hp.model_type)
    return model
