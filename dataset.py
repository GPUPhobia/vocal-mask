import numpy as np

import os

import torch
from torch.utils.data import DataLoader, Dataset
from hparams import hparams as hp
from tqdm import tqdm

class SpectrogramDataset(Dataset):
    def __init__(self, data_path, spec_info):
        self.data_path = data_path
        self.mix_path = os.path.join(data_path, "mix")
        self.vox_path = os.path.join(data_path, "vox")
        self.offset = hp.stft_frames//2
        self.metadata = self.get_slices(spec_info)

    def get_slices(self, spec_info):
        metadata = []
        print("Preparing dataset")
        for spec in tqdm(spec_info):
            size = spec[1] - hp.stft_frames
            fname = spec[0]+".npy"
            for i in range(0, size, hp.stft_stride):
                j = i + hp.stft_frames
                slice_info = (spec[0], i, j)
                metadata.append(slice_info)
        return metadata

    def __getitem__(self, index):
        slice_info = self.metadata[index]
        fname = slice_info[0]+".npy"
        i = slice_info[1]
        j = slice_info[2]
        x = np.load(os.path.join(self.mix_path, fname), mmap_mode='r')[:,:,i:j]
        y = np.load(os.path.join(self.vox_path, fname), mmap_mode='r')[:,i+self.offset]
        return x, y

    def __len__(self):
        return len(self.metadata)


def basic_collate(batch):
    x = [it[0] for it in batch]
    x = np.stack(x).astype(np.float32)
    x = torch.FloatTensor(x)
    y = [it[1] for it in batch]
    y = np.stack(y)
    if hp.mask_threshold is not None:
        y = y > hp.mask_threshold
    y = torch.FloatTensor(y.astype(np.float32))
    return x, y
