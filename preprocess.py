"""
Preprocess dataset

usage: preprocess.py [options] <mix-dir> <vox-dir>

options:
     --output-dir=<dir>      Directory where processed outputs are saved. [default: data_dir].
    -h, --help              Show help message.
"""
import os
from docopt import docopt
import numpy as np
import math, pickle, os
from audio import *
from hparams import hparams as hp
from utils import *
from tqdm import tqdm
import pickle


def process_data(mix_dir, vox_dir, output_path, mix_path, vox_path): 
    dataset_ids = []
    mix_wavf = os.listdir(mix_dir)
    vox_wavf = os.listdir(vox_dir)
    count = len(mix_wavf)
    for i in tqdm(range(count)):
        file_id = f"{i:06d}"

        # convert wav to spectrogram for mixture
        mix_wav = load_wav(os.path.join(mix_dir, mix_wavf[i]))
        mix_spec = melspectrogram(mix_wav)
        mix_spec = mix_spec[np.newaxis,:,:] # single channel
        
        # convert wav to spectrogram for vocal
        vox_wav = load_wav(os.path.join(vox_dir, vox_wavf[i]))
        vox_spec = melspectrogram(vox_wav)
        width = vox_spec.shape[1]
        # we only want to predict the middle frame of vocal spectrogram
        vox_spec = vox_spec[:,width//2]

        # save output
        dataset_ids.append(file_id)
        np.save(os.path.join(mix_path,file_id+".npy"), mix_spec)
        np.save(os.path.join(vox_path,file_id+".npy"), vox_spec)
    
    with open(os.path.join(output_path,'dataset_ids.pkl'), 'wb') as f:
        pickle.dump(dataset_ids, f)


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

