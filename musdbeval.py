"""Extract vocals from waveform.

usage: musdbeval.py [options] <checkpoint-path> <musdb-root>

options:
    -h, --help                  Show this help message and exit
"""
from docopt import docopt

import os
from os.path import dirname, join, expanduser
import random
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pickle

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
import pickle

from audio import *
from model import build_model
from hparams import hparams as hp
import musdb
import museval
from utils import pad_audio, resample, resize, replicate_channels

use_cuda = torch.cuda.is_available()
device = None
model = None

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

def evaluate(track):
    mix_audio, orig_sr, mix_channels = track.audio, track.rate, track.audio.shape[1]
    if mix_channels > 1:
        mono_audio = librosa.to_mono(mix_audio.T)
    else:
        mono_audio = mix_audio
    mono_audio = pad_audio(mono_audio, orig_sr)
    if orig_sr != hp.sample_rate:
        mono_audio = librosa.resample(mono_audio, orig_sr, hp.sample_rate)
    estimates = model.generate_wav(device, mono_audio)
    if hp.sample_rate != orig_sr:
        resample(estimates, orig_sr)
    resize(estimates, mix_audio)
    if mix_channels > 1:
        replicate_channels(estimates, mix_channels)
    #scores = museval.eval_mus_track(
    #    track, estimates, output_dir='bss_evals')
    #print(scores)
    return estimates


if __name__=="__main__":
    args = docopt(__doc__)
    #print("Command line args:\n", args)
    checkpoint_path = args["<checkpoint-path>"]
    musdb_dir = args["<musdb-root>"]

    device = torch.device("cuda" if use_cuda else "cpu")
    print("using device:{}".format(device))

    # build model
    model = build_model().to(device)

    # load checkpoint
    model = load_checkpoint(checkpoint_path, model)
    print("loading model from checkpoint:{}".format(checkpoint_path))

    mus = musdb.DB(root_dir=musdb_dir, is_wav=True)
    mus.run(evaluate, subsets="test", estimates_dir="estimates")

    museval.eval_mus_dir(dataset=mus, estimates_dir="estimates", output_dir="bss_evals", subsets="test", parallel=True, is_wav=True)
