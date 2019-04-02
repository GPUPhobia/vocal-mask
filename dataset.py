import numpy as np

import os
import random

import torch
from torch.utils.data import DataLoader, Dataset
from hparams import hparams as hp
from utils import mulaw_quantize, inv_mulaw_quantize
import pickle

class SpectrogramDataset(Dataset):
    def __init__(self, data_path, dataset_ids_fname):
        self.path = os.path.join(data_path, "")
        with open(os.path.join(self.path, dataset_ids_fname), 'rb') as f:
            self.metadata = pickle.load(f)
        self.mix_path = os.path.join(data_path, "mix")
        self.vox_path = os.path.join(data_path, "vox")

    def __getitem__(self, index):
        file_id = self.metadata[index]
        x = np.load(os.path.join(self.mix_path, f"{file_id}.npy"))
        y = np.load(os.path.join(self.vox_path, f"{file_id}.npy"))
        # select the middle stft frame as the label
        return x, y

    def __len__(self):
        return len(self.metadata)


def scale_vector(arr, low, high):
    return np.interp(arr, (arr.min(), arr.max()), (low, high))

def basic_collate(batch):
    x = [it[0] for it in batch]
    x = np.stack(x).astype(np.float32)
    x = torch.FloatTensor(x)
    y = [scale_vector(it[1], 0, 1) for it in batch]
    y = np.stack(y).astype(np.float32)
    y = torch.FloatTensor(y)
    return x, y
