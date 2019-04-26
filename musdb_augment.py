"""
Augment the musdb dataset

usage: python musdb_augment.py [options] <musdb-root>

options:
    -h, --help              Show help message.
"""

import musdb
import librosa
import librosa.output
import librosa.effects
import numpy as np
import os
#from docopt import docopt
import sys
from tqdm import tqdm

def pitch_shift(wav, steps):
    sr = 44100
    wav = librosa.to_mono(wav.T)
    return librosa.effects.pitch_shift(wav, sr, n_steps=steps)

def save_tracks(mixture, vocals, path):
    sr = 44100
    mixture = np.clip(mixture, -0.999, 0.999)
    os.makedirs(path, exist_ok=True)
    librosa.output.write_wav(os.path.join(path, "mixture.wav"), mixture, sr)
    librosa.output.write_wav(os.path.join(path, "vocals.wav"), vocals, sr)


if __name__=="__main__":
    #args = docopt(__doc__)
    #root_dir = args["<musdb-root>"]
    root_dir = sys.argv[1]
    mus = musdb.DB(root_dir=root_dir, is_wav=True)
    tracks = mus.load_mus_tracks(subsets=["train"])
    sr = 44100
    out_dir = os.path.join(root_dir, "augment")
    for track in tqdm(tracks):
        name = track.name
        vocals = track.targets["vocals"].audio
        acc = track.targets["accompaniment"].audio
        acc = librosa.to_mono(acc.T)

        yvocals = pitch_shift(vocals, 3)
        mixture = acc + yvocals
        path = os.path.join(out_dir, name+"_up")
        save_tracks(mixture, yvocals, path)

        yvocals = pitch_shift(vocals, -3)
        mixture = acc + yvocals
        path = os.path.join(out_dir, name+"_down")
        save_tracks(mixture, yvocals, path)
