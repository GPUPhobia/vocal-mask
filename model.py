import torch
from torch import nn
import torch.nn.functional as F
from hparams import hparams as hp
from torch.utils.data import DataLoader, Dataset
from distributions import *
from utils import num_params, mulaw_quantize, inv_mulaw_quantize

from tqdm import tqdm
import numpy as np

class Model(nn.Module):
    def __init__(self, input_dims, output_dims):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dims, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 64, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 16, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(16)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout(p=0.25)
        self.dropout2 = nn.Dropout(p=0.50)
        self.fc1 = nn.Linear(1344, 64)
        self.bn5 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, output_dims)
        num_params(self)
    
    def forward(self, x):
        bsize = x.size(0)
        x = self.bn1(F.leaky_relu(self.conv1(x)))
        x = self.bn2(F.leaky_relu(self.conv2(x)))
        x = self.maxpool1(x)
        x = self.dropout1(x)
        x = self.bn3(F.leaky_relu(self.conv3(x)))
        x = self.bn4(F.leaky_relu(self.conv4(x)))
        x = self.maxpool2(x)
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)
        x = self.bn5(F.leaky_relu(self.fc1(x)))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

    def generate(self, wav):
        """Given a waveform, generate the vocal-only spectrogram slices.
        Another network will need to convert the spectrogram slices back
        into waveforms that can then be concatenated"""

        pass

def build_model():
    model = Model(1, hp.num_mels)
    return model
