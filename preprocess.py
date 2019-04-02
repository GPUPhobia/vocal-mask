"""
Preprocess dataset

usage: preprocess.py [options] <mix-dir> <vox-dir>

options:
     --output-dir=<dir>      Directory where processed outputs are saved. [default: data_dir].
    -h, --help              Show help message.
"""
import os
import random
from docopt import docopt
import numpy as np
import math, pickle, os
from audio import *
from hparams import hparams as hp
from utils import *
from tqdm import tqdm
import librosa
import librosa.feature
import random

def get_mel(path, preemp):
    wav = load_wav(path)
    if preemp:
        wav = preemphasis(wav)
    return librosa.feature.melspectrogram(y=wav, sr=hp.sample_rate,
        n_fft=hp.fft_size, hop_length=hp.hop_size, n_mels=hp.num_mels)

def process_data(mix_dir, vox_dir, output_path, mix_path, vox_path): 
    dataset_ids = []
    mix_wavf = os.listdir(mix_dir)
    vox_wavf = os.listdir(vox_dir)
    count = len(mix_wavf)
    for i in tqdm(range(count)):
        file_id = f"{i:06d}"
        mix_mel = get_mel(os.path.join(mix_dir, mix_wavf[i]), 
                          hp.use_preemphasis)
        mix_mel = [np.newaxis,:,:] # single channel
        vox_mel = get_mel(os.path.join(vox_dir, vox_wavf[i]), 
                          hp.use_preemphasis)
        # we only want to predict the middle frame of vocal spectrogram
        vox_mel_slice = vox_mel[:,hp.stft_frames//2]
        dataset_ids.append(file_id)
        np.save(os.path.join(mix_path,file_id+".npy"), mix_mel)
        np.save(os.path.join(vox_path,file_id+".npy"), vox_mel_slice)
    
    random.shuffle(dataset_ids)
    split = int(count*hp.train_test_split)
    testset_ids = dataset_ids[:split]
    trainset_ids = dataset_ids[split:]
    with open(os.path.join(output_path,'trainset_ids.pkl'), 'wb') as f:
        pickle.dump(trainset_ids, f)
    with open(os.path.join(output_path,'testset_ids.pkl'), 'wb') as f:
        pickle.dump(testset_ids, f)


if __name__=="__main__":
    args = docopt(__doc__)
    mix_dir = args["<mix-dir>"]
    vox_dir = args["<vox-dir>"]
    output_dir = args["--output-dir"]

    # create paths
    output_path = os.path.join(output_dir,"")
    mix_path = os.path.join(output_path, "mix")
    vox_path = os.path.join(output_path, "vox")

    # create dirs
    os.makedirs(mix_path, exist_ok=True)
    os.makedirs(vox_path, exist_ok=True)

    # process data
    process_data(mix_dir, vox_dir, output_path, mix_path, vox_path)

