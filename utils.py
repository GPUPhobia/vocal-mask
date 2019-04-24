import librosa
import numpy as np
import torch
from hparams import hparams as hp

def num_params(model) :
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1000000
    print('Trainable Parameters: %.3f million' % parameters)

def pad_audio(audio, sr):
    hop_len = (sr//hp.sample_rate)*hp.hop_size
    left_over = hop_len - audio.shape[0]%hop_len
    return np.pad(audio, (0, left_over), 'constant', constant_values=0)

def resample(estimates, sr):
    for key in estimates:
        estimates[key] = librosa.resample(estimates[key], hp.sample_rate, sr)

def resize(estimates, mix_audio):
    target_len = mix_audio.shape[0]
    for key in estimates:
        estimates[key] = estimates[key][:target_len]

def replicate_channels(estimates, mix_channels):
    for key in estimates:
        estimates[key] = np.tile(estimates[key], (mix_channels,1)).T
