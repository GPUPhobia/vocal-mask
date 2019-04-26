"""
Preprocess dataset

usage: build_dataset.py [options] <root-dir> <output-dir>

options:
    --parser=<parser_type>  Type of dataset parser (dsd or musdb, default dsd)
    -h, --help              Show help message.
"""

from audio import *
import os
import os.path
from hparams import hparams as hp
from tqdm import tqdm
import librosa
import musdb
from dsdparser import DSD
import random
import numpy as np
import pickle
from docopt import docopt

# 2 args, the musdb root directory, and the directory to save the output

args = docopt(__doc__)
root_dir = args["<root-dir>"]
output_dir = args["<output-dir>"]
parser_type = args["--parser"]
if parser_type is None:
    parser_type = 'dsd'


if parser_type == 'musdb':
    dataset = musdb.DB(root_dir=root_dir, is_wav=True)
    tracks = dataset.load_mus_tracks(subsets=['train', 'augment'])
    eval_tracks = dataset.load_mus_tracks(subsets=['test'])
else:
    dataset = DSD(root_dir=root_dir)
    tracks = dataset.load_tracks()
    random.shuffle(tracks)
    eval_tracks = tracks[:8]
    tracks = tracks[8:]

os.makedirs(output_dir, exist_ok=True)
mixture_path = os.path.join(output_dir, "mix")
vocal_path = os.path.join(output_dir, "vox")
eval_path = os.path.join(output_dir, "eval")
eval_mix_path = os.path.join(eval_path, "mix")
eval_vox_path = os.path.join(eval_path, "vox")
test_path = os.path.join(output_dir, "test")
test_mix_path = os.path.join(test_path, "mix")
test_vox_path = os.path.join(test_path,"vox")
os.makedirs(mixture_path, exist_ok=True)
os.makedirs(vocal_path, exist_ok=True)
os.makedirs(eval_mix_path, exist_ok=True)
os.makedirs(eval_vox_path, exist_ok=True)
os.makedirs(test_mix_path, exist_ok=True)
os.makedirs(test_vox_path, exist_ok=True)

def load_musdb_sample(track):
    audio = track.audio
    sample_rate = 44100
    #sample_rate = track.rate
    if len(audio.shape) == 2:
        audio = librosa.to_mono(audio.T)
    if sample_rate != hp.sample_rate:
        audio = librosa.resample(audio, sample_rate, hp.sample_rate)
    return audio

def load_dsd_sample(track):
    return track.load()

def load_samples(track):
    vocal_track = track.targets['vocals']
    if parser_type == 'musdb':
        mixture = load_musdb_sample(track)
        vocal = load_musdb_sample(vocal_track)
    else:
        mixture = load_dsd_sample(track)
        vocal = load_dsd_sample(vocal_track)
    return mixture, vocal
    
spec_info = []
print("processing training samples")
for idx, track in enumerate(tqdm(tracks)):
    mixture, vocal = load_samples(track)
    mix_spec = spectrogram(mixture, power=hp.mix_power_factor)[0][np.newaxis,:,:]
    vox_spec = spectrogram(vocal, power=hp.vox_power_factor)[0]
    spec_id = f"spec{idx:06d}"
    spec_info.append((spec_id, mix_spec.shape[2]))
    np.save(os.path.join(mixture_path, spec_id+".npy"), mix_spec)
    np.save(os.path.join(vocal_path, spec_id+".npy"), vox_spec)
with open(os.path.join(output_dir, 'spec_info.pkl'), 'wb') as f:
    random.shuffle(spec_info)
    pickle.dump(spec_info, f)

test_spec_info = []
print("processing validation samples")
for idx, track in enumerate(tqdm(eval_tracks)):
    mixture, vocal = load_samples(track)
    mix_spec = spectrogram(mixture, power=hp.mix_power_factor)[0][np.newaxis,:,:]
    vox_spec = spectrogram(vocal, power=hp.vox_power_factor)[0]
    spec_id = f"spec{idx:06d}"
    test_spec_info.append((spec_id, mix_spec.shape[2]))
    np.save(os.path.join(test_mix_path, spec_id+".npy"), mix_spec)
    np.save(os.path.join(test_vox_path, spec_id+".npy"), vox_spec)
with open(os.path.join(test_path, 'test_spec_info.pkl'), 'wb') as f:
    random.shuffle(test_spec_info)
    pickle.dump(test_spec_info, f)

i = 0
print("slicing eval samples")
for track in tqdm(eval_tracks):
    mixture, vocal = load_samples(track)
    slices = get_wav_slices(mixture, hp.eval_length, hp.eval_length)
    for j,k in slices:
        # skip slices with no audio content
        if np.sum(vocal[j:k]) == 0:
            continue
        fname = f"{i:06d}.wav"
        save_wav(mixture[j:k], os.path.join(eval_mix_path, fname))
        save_wav(vocal[j:k], os.path.join(eval_vox_path, fname))
        i += 1
