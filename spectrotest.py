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

PREEMPHASIS = False
SPEC_TYPE = 'mel'
MASK_TSFM = 'linear'

mixfs = os.listdir(mix_dir)
voxfs = os.listdir(vox_dir)

def get_wav(path, preemp=True):
    wav = load_wav(path)
    if preemp:
        wav = preemphasis(wav)
    return wav

def get_spec(wav, spec_type='mel'):
    if spec_type == 'mel':
        #return librosa.feature.melspectrogram(wav, sr=hp.sample_rate,
        #    n_fft=hp.fft_size, hop_length=hp.hop_size, n_mels=hp.num_mels)
        return melspectrogram(wav)
    elif spec_type == 'if':
        return librosa.ifgram(wav, sr=hp.sample_rate, n_fft=hp.fft_size,
            hop_length=hp.hop_size)[0]
    elif spec_type == 'stft':
        #return np.abs(librosa.stft(wav, n_fft=hp.fft_size, hop_length=hp.hop_size))
        return spectrogram(wav)
    else:
        raise ValueError(f"Unknown spec_type: `{spec_type}`")

def show_spec(spec):
    if SPEC_TYPE == 'mel':
        #spec = librosa.power_to_db(spec, ref=np.max)
        y_scale = 'mel'
    else:
        #spec = librosa.amplitude_to_db(spec, ref=np.max)
        y_scale = 'mel'
    librosa.display.specshow(spec, y_axis=y_scale, x_axis='time')

def sigmoid(x):
    #z = np.exp(x-np.max(x))
    #return z/(1+z)
    return scipy.special.expit(x)

mixf = mixfs[idx]
voxf = voxfs[idx]
mix_wav = get_wav(os.path.join(mix_dir, mixf), preemp=PREEMPHASIS)
mix_mel = get_spec(mix_wav, spec_type=SPEC_TYPE)
vox_wav = get_wav(os.path.join(vox_dir, voxf), preemp=PREEMPHASIS)
vox_mel = get_spec(vox_wav, spec_type=SPEC_TYPE)

# mask vocal spec onto mixture spec
if MASK_TSFM == 'sqrt':
    vox_mask = np.sqrt(vox_mel)
    vox_max = np.sqrt(mix_mel).max()
    vox_min = vox_mask.min()
else:
    vox_mask = vox_mel
    vox_max = mix_mel.max()
    vox_min = vox_mask.min()

#mask = vox_mask > 0.5
avg = np.sum(mix_mel)/vox_mask.size
print(avg)
mask = vox_mask > avg
new_mel = mix_mel * mask

# Show spectrograms
plt.figure(figsize=(12,8))
plt.subplot(131)
plt.title('Mixture Spectrogram')
show_spec(mix_mel)
plt.colorbar(format='%+2.0f dB')

plt.subplot(132)
plt.title('Vocal Spectrogram')
show_spec(vox_mel)
plt.colorbar(format='%+2.0f dB')

plt.subplot(133)
plt.title('Masked Mixture Spectrogram')
show_spec(new_mel)
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()
