"""
Augment the musdb dataset

usage: python musdb_augment.py [options] <musdb-root>

options:
    -h, --help              Show help message.
"""

import musdb
import librosa
import librosa.output
import numpy as np
import os
#from docopt import docopt
import sys
from tqdm import tqdm

if __name__=="__main__":
    #args = docopt(__doc__)
    #root_dir = args["<musdb-root>"]
    root_dir = sys.argv[1]
    mus = musdb.DB(root_dir=root_dir, is_wav=True)
    tracks = mus.load_mus_tracks(subsets=["train"])
    sr = 44100
    for track in tqdm(tracks):
        name = track.name
        vocals = track.targets["vocals"].audio
        acc = track.targets["accompaniment"].audio
        new_acc = acc[sr*10:,:]
        new_drums = track.targets["drums"].audio[sr*10:,:]
        new_bass = track.targets["bass"].audio[sr*10:,:]
        new_other = track.targets["other"].audio[sr*10:,:]
        new_vocals = vocals[:-sr*10,:]
        new_mixture = new_acc + new_vocals
        new_mixture = np.clip(new_mixture, -0.999, 0.999)
        new_name = name + "_tsfm"
        new_dir = os.path.join(root_dir, "train", new_name)
        os.makedirs(new_dir, exist_ok=True)
        librosa.output.write_wav(os.path.join(new_dir, "mixture.wav"), new_mixture, sr)
        librosa.output.write_wav(os.path.join(new_dir, "vocals.wav"), new_vocals, sr)
        librosa.output.write_wav(os.path.join(new_dir, "drums.wav"), new_drums, sr)
        librosa.output.write_wav(os.path.join(new_dir, "bass.wav"), new_bass, sr)
        librosa.output.write_wav(os.path.join(new_dir, "other.wav"), new_other, sr)
