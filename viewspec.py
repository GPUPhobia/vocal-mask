from audio import *
import os
import os.path
import sys
import librosa
import librosa.feature
import librosa.display
import scipy
import scipy.special
import numpy as np
from hparams import hparams as hp
import platform

import matplotlib
if platform.system() == 'Darwin':
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

mix_dir = sys.argv[1]
vox_dir = sys.argv[2]
idx = int(sys.argv[3])


mixfs = os.listdir(mix_dir)
voxfs = os.listdir(vox_dir)
start = hp.hop_size*hp.stft_frames*40 + 24
end = start + hp.hop_size*200 - 24

def get_wav(path):
    wav = load_wav(path)
    return wav[start:end]

def get_spec(wav):
    return melspectrogram(wav)

def show_spec(spec):
    librosa.display.specshow(spec, y_axis='mel', x_axis='time')

mixf = next(f for f in mixfs if int(f[:3]) == idx)
voxf = next(f for f in voxfs if int(f[:3]) == idx)
mix_wav = get_wav(os.path.join(mix_dir, mixf, 'mixture.wav'))
mix_mel = get_spec(mix_wav)
vox_wav = get_wav(os.path.join(vox_dir, voxf, 'vocals.wav'))
vox_mel = get_spec(vox_wav)

# Show spectrograms
plt.figure()
plt.subplot(121)
plt.title('Mixture Spectrogram')
show_spec(mix_mel)

plt.subplot(122)
plt.title('Vocal Spectrogram')
show_spec(vox_mel)

plt.tight_layout()
plt.show()
