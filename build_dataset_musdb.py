from audio import *
import os
import os.path
import sys
from hparams import hparams as hp
from tqdm import tqdm
import librosa
import musdb

# slices the musdb dataset into slices based on `window` and `stride`
# 2 args, the musdb root directory, and the directory to save the output

window = hp.hop_size*hp.stft_frames-1
stride = hp.hop_size*hp.stft_stride

musdb_root_dir = sys.argv[1]
output_dir = sys.argv[2]
mus = musdb.DB(root_dir=musdb_root_dir)
tracks = mus.load_mus_tracks(subsets=['train'])
os.makedirs(output_dir, exist_ok=True)
mixture_path = os.path.join(output_dir, "mixture")
vocal_path = os.path.join(output_dir, "vocal")
os.makedirs(mixture_path, exist_ok=True)
os.makedirs(vocal_path, exist_ok=True)


ids = []

i = 0
print("slicing samples")
for track in tqdm(tracks):
    mixture = track.audio
    vocal = track.targets['vocals'].audio
    sample_rate = track.rate
    mix_wav = librosa.to_mono(mixture.T)
    vox_wav = librosa.to_mono(vocal.T)
    if sample_rate != hp.sample_rate:
        mix_wav = librosa.resample(mix_wav, sample_rate, hp.sample_rate)
        vox_wav = librosa.resample(vox_wav, sample_rate, hp.sample_rate)
    slices = get_wav_slices(mix_wav, window, stride)
    for j,k in slices:
        # skip slices with no audio content
        if np.sum(mix_wav[j:k]) == 0:
            continue
        fname = f"{i:06d}.wav"
        save_wav(mix_wav[j:k], os.path.join(mixture_path, fname))
        save_wav(vox_wav[j:k], os.path.join(vocal_path, fname))    
        i += 1
