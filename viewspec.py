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
fname = sys.argv[2]

mix_dir = os.path.join(eval_dir, 'mix')
vox_dir = os.path.join(eval_dir, 'vox')

mix_wav = load_wav(os.path.join(mix_dir, fname))
mix_mel, mix_spec = spectrogram(mix_wav, power=hp.mix_power_factor)
vox_wav = load_wav(os.path.join(vox_dir, fname))
vox_mel, vox_spec = spectrogram(vox_wav, power=hp.vox_power_factor)
vox_mask = vox_mel > hp.mask_threshold
gen_vox = mix_spec*vox_mask
wav = inv_spectrogram(gen_vox)
save_wav(wav, 'test.wav')
print("Masked waveform saved to `test.wav`")

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
