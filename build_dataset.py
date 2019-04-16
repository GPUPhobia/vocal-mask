from audio import *
import os
import os.path
import sys
from hparams import hparams as hp
from tqdm import tqdm
import librosa
import musdb
from dsdparser import DSD
import random

# slices the musdb or dsd dataset into slices based on `window` and `stride`
# 2 args, the musdb root directory, and the directory to save the output

#PARSER_TYPE = 'musdb'
PARSER_TYPE = 'dsd'
window = hp.hop_size*hp.stft_frames-1
stride = hp.hop_size*hp.stft_stride

root_dir = sys.argv[1]
output_dir = sys.argv[2]

if PARSER_TYPE == 'musdb':
    dataset = musdb.DB(root_dir=root_dir)
    tracks = dataset.load_mus_tracks(subsets=['train'])
    eval_tracks = dataset.load_mus_tracks(subsets=['test'])
else:
    dataset = DSD(root_dir=root_dir)
    tracks = dataset.load_tracks()
    random.shuffle(tracks)
    eval_tracks = tracks[:8]
    tracks = tracks[8:]

os.makedirs(output_dir, exist_ok=True)
mixture_path = os.path.join(output_dir, "mixture")
vocal_path = os.path.join(output_dir, "vocal")
eval_path = os.path.join(output_dir, "eval")
eval_mix_path = os.path.join(eval_path, "mixture")
eval_vox_path = os.path.join(eval_path, "vocal")
os.makedirs(mixture_path, exist_ok=True)
os.makedirs(vocal_path, exist_ok=True)
os.makedirs(eval_mix_path, exist_ok=True)
os.makedirs(eval_vox_path, exist_ok=True)

def load_musdb_sample(track):
    audio = track.audio
    sample_rate = track.rate
    audio = librosa.to_mono(audio.T)
    if sample_rate != hp.sample_rate:
        audio = librosa.resample(audio, sample_rate, hp.sample_rate)
    return audio

def load_dsd_sample(track):
    return track.load()

def load_samples(track):
    vocal_track = track.targets['vocals']
    if PARSER_TYPE == 'musdb':
        mixture = load_musdb_sample(track)
        vocal = load_musdb_sample(vocal_track)
    else:
        mixture = load_dsd_sample(track)
        vocal = load_dsd_sample(vocal_track)
    return mixture, vocal
    
i = 0
print("slicing training samples")
for track in tqdm(tracks):
    mixture, vocal = load_samples(track)
    slices = get_wav_slices(mixture, window, stride)
    for j,k in slices:
        # skip slices with no audio content
        if np.sum(mixture[j:k]) == 0:
            continue
        fname = f"{i:06d}.wav"
        save_wav(mixture[j:k], os.path.join(mixture_path, fname))
        save_wav(vocal[j:k], os.path.join(vocal_path, fname))    
        i += 1

i = 0
print("slicing eval samples")
for track in tqdm(eval_tracks):
    mixture, vocal = load_samples(track)
    slices = get_wav_slices(mixture, hp.eval_length, hp.eval_length)
    for j,k in slices:
        # skip slices with no audio content
        if np.sum(mixture[j:k]) == 0:
            continue
        fname = f"{i:06d}.wav"
        save_wav(mixture[j:k], os.path.join(eval_mix_path, fname))
        save_wav(vocal[j:k], os.path.join(eval_vox_path, fname))
        i += 1
