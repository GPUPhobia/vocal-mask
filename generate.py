"""Extract vocals from waveform.

usage: generate.py [options] <checkpoint-path> <input-wav>

options:
    --output-dir=<dir>      Directory where to save output wav [default: generated].
    --sr                    Sample rate of generated waveform
    -h, --help                  Show this help message and exit
"""
from docopt import docopt

import os
from os.path import dirname, join, expanduser
import random
from tqdm import tqdm

import numpy as np
import librosa
import librosa.display

import torch

from audio import *
from model import build_model
from hparams import hparams as hp
from utils import resample

use_cuda = torch.cuda.is_available()

def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def load_checkpoint(path, model):
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    model.load_state_dict(checkpoint["state_dict"])
    return model


def generate(device, model, path, output_dir, target_sr):
    wav = load_wav(path)
    estimates = model.generate_wav(device, wav)
    if target_sr != hp.sample_rate:
        resample(estimates, target_sr)
    file_id = path.split('/')[-1].split('.')[0]
    vox_outpath = os.path.join(output_dir, f'{file_id}_vocals.wav')
    bg_outpath = os.path.join(output_dir, f'{file_id}_accompaniment.wav')
    save_wav(estimates['vocals'], vox_outpath)
    save_wav(estimates['accompaniment'], bg_outpath)
    

if __name__=="__main__":
    args = docopt(__doc__)
    output_dir = args["--output-dir"]
    checkpoint_path = args["<checkpoint-path>"]
    input_path = args["<input-wav>"]
    target_sr = args["--sr"]

    if output_dir is None:
        output_dir = 'generated'
    if target_sr is None:
        target_sr = hp.sample_rate

    # make dirs, load dataloader and set up device
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if use_cuda else "cpu")
    print("using device:{}".format(device))

    # build model, create optimizer
    model = build_model().to(device)

    # load checkpoint
    model = load_checkpoint(checkpoint_path, model)
    print("loading model from checkpoint:{}".format(checkpoint_path))

    generate(device, model, input_path, output_dir, target_sr)
