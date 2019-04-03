from audio import *
import os
import os.path
import sys
import librosa
import librosa.feature
import librosa.display
import scipy
import scipy.interpolate
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

PREEMPHASIS = True
SPEC_TYPE = 'mel'
POWER = True
Y_SCALE = 'mel'

mixfs = os.listdir(mix_dir)
voxfs = os.listdir(vox_dir)

def get_wav(path, preemp=True):
    wav = load_wav(path)
    if preemp:
        wav = preemphasis(wav)
    return wav

def get_spec(wav, spec_type='mel'):
    if spec_type == 'mel':
        return librosa.feature.melspectrogram(wav, sr=hp.sample_rate,
            n_fft=hp.fft_size, hop_length=hp.hop_size, n_mels=hp.num_mels)
    elif spec_type == 'if':
        return librosa.ifgram(wav, sr=hp.sample_rate, n_fft=hp.fft_size,
            hop_length=hp.hop_size)[0]
    else:
        raise ValueError(f"Unknown spec_type: `{spec_type}`")

def show_spec(spec, power_to_db=True, y_scale='mel'):
    if power_to_db:
        spec = librosa.power_to_db(spec, ref=np.max)
    librosa.display.specshow(spec, y_axis=y_scale, x_axis='time')

mixf = mixfs[idx]
voxf = voxfs[idx]
mix_wav = get_wav(os.path.join(mix_dir, mixf), preemp=PREEMPHASIS)
mix_mel = get_spec(mix_wav, spec_type=SPEC_TYPE)
vox_wav = get_wav(os.path.join(vox_dir, voxf), preemp=PREEMPHASIS)
vox_mel = get_spec(vox_wav, spec_type=SPEC_TYPE)

# mask vocal spec onto mixture spec
vox_min = vox_mel.min()
vox_max = vox_mel.max()
scaled = np.interp(vox_mel, (vox_min, vox_max), (0, 1))
new_mel = mix_mel * scaled

# Show spectrograms
plt.figure(figsize=(18,16))
plt.subplot(131)
plt.title('Mixture Spectrogram')
show_spec(mix_mel, power_to_db=POWER, y_scale=Y_SCALE)
plt.colorbar(format='%+2.0f dB')

plt.subplot(132)
plt.title('Vocal Spectrogram')
show_spec(vox_mel, power_to_db=POWER, y_scale=Y_SCALE)
plt.colorbar(format='%+2.0f dB')

plt.subplot(133)
plt.title('Masked Mixture Spectrogram')
show_spec(new_mel, power_to_db=POWER, y_scale=Y_SCALE)
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()
