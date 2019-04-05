import numpy as np

import os

import torch
from torch.utils.data import DataLoader, Dataset
from hparams import hparams as hp
from utils import mulaw_quantize, inv_mulaw_quantize

class SpectrogramDataset(Dataset):
    def __init__(self, data_path, dataset_ids):
        self.path = os.path.join(data_path, "")
        self.metadata = dataset_ids
        self.mix_path = os.path.join(data_path, "mix")
        self.vox_path = os.path.join(data_path, "vox")

    def __getitem__(self, index):
        file_id = self.metadata[index]
        x = np.load(os.path.join(self.mix_path, f"{file_id}.npy"))
        y = np.load(os.path.join(self.vox_path, f"{file_id}.npy"))
        return x, (y > 0.5)

    def __len__(self):
        return len(self.metadata)


def basic_collate(batch):
    x = [it[0] for it in batch]
    x = np.stack(x).astype(np.float32)
    x = torch.FloatTensor(x)
    y = [scale_vector(it[1], 0, 0, 1) for it in batch]
    y = np.stack(y).astype(np.float32)
    y = torch.FloatTensor(y)
    return x, y
