from audio import *
import os
import os.path
import sys
from hparams import hparams as hp
from tqdm import tqdm

input_dir = sys.argv[1]
target_dir = sys.argv[2]
output_dir = sys.argv[3]
input_files = os.listdir(input_dir)
target_files = os.listdir(target_dir)
os.makedirs(output_dir, exist_ok=True)
mixture_path = os.path.join(output_dir, "mixture")
vocal_path = os.path.join(output_dir, "vocal")
os.makedirs(mixture_path, exist_ok=True)
os.makedirs(vocal_path, exist_ok=True)
i = 0
window = hp.hop_size*hp.stft_frames-1
stride = hp.hop_size*hp.stft_stride
print("slicing mixture samples")
for f in tqdm(input_files):
    wav = load_wav(os.path.join(input_dir, f, "mixture.wav"))
    slices = get_wav_slices(wav, window, stride)
    for j,k in slices:
        fname = f"{i:06d}.wav"
        save_wav(wav[j:k], os.path.join(mixture_path, fname))
        i += 1
i = 0
print("slicing vocal samples")
for f in tqdm(target_files):
    wav = load_wav(os.path.join(target_dir, f, "vocals.wav"))
    slices = get_wav_slices(wav, window, stride)
    for j,k in slices:
        fname = f"{i:06d}.wav"
        save_wav(wav[j:k], os.path.join(vocal_path, fname))
        i += 1
