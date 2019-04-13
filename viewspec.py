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

eval_dir = sys.argv[1]
idx = int(sys.argv[2])

mix_dir = os.path.join(eval_dir, 'mix')
vox_dir = os.path.join(eval_dir, 'vox')

def get_wav(path):
    wav = load_wav(path)
    return wav

def get_spec(wav):
    return melspectrogram(wav)

fname = f"eval{idx:06d}.wav"
mix_wav = get_wav(os.path.join(mix_dir, fname))
mix_mel = get_spec(mix_wav)
vox_wav = get_wav(os.path.join(vox_dir, fname))
vox_mel = get_spec(vox_wav)
vox_mask = vox_mel > hp.mask_threshold

# Show spectrograms
plt.figure()
plt.subplot(221)
plt.title('Mixture Spectrogram')
show_spec(mix_mel)

plt.subplot(222)
plt.title('Vocal Spectrogram')
show_spec(vox_mel)

plt.subplot(223)
plt.title('Vocal Mask')
show_spec(vox_mask)

plt.subplot(224)
plt.title('Masked Mixture')
show_spec(mix_mel*vox_mask)

plt.tight_layout()
plt.show()
