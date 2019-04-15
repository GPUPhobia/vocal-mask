"""Training WaveRNN Model.

usage: train.py [options] <checkpoint-path> <input-wav>

options:
    --output-dir=<dir>      Directory where to save output wav [default: generated].
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

from audio import *
from model import build_model
from loss_function import nll_loss
from dataset import basic_collate, SpectrogramDataset
from hparams import hparams as hp
from lrschedule import noam_learning_rate_decay, step_learning_rate_decay

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


def generate(device, model, path, output_dir):
    wav = load_wav(path)
    y = model.generate(device, wav)
    file_id = path.split('/')[-1].split('.')[0]
    outpath = os.path.join(output_dir, f'generated_{file_id}.wav')
    save_wav(y, outpath)
    

if __name__=="__main__":
    args = docopt(__doc__)
    #print("Command line args:\n", args)
    output_dir = args["--output-dir"]
    checkpoint_path = args["<checkpoint-path>"]
    input_path = args["<input-wav>"]

    if output_dir is None:
        output_dir = 'generated'

    # make dirs, load dataloader and set up device
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if use_cuda else "cpu")
    print("using device:{}".format(device))

    # build model, create optimizer
    model = build_model().to(device)

    # load checkpoint
    model = load_checkpoint(checkpoint_path, model)
    print("loading model from checkpoint:{}".format(checkpoint_path))

    generate(device, model, input_path, output_dir)
