import librosa
import librosa.effects
import librosa.feature
import librosa.filters
import math
import numpy as np
from scipy import signal
from hparams import hparams
from scipy.io import wavfile


def load_wav(path):
    wav = librosa.load(path, sr=hparams.sample_rate)[0]
    return wav

def get_wav_slices(wav, window, stride):
    N = len(wav)
    return [(i,i+window) for i in range(0, N-window, stride)]

def save_wav(wav, path):
    wav = wav * 32767 / max(0.01, np.max(np.abs(wav)))
    wavfile.write(path, hparams.sample_rate, wav.astype(np.int16))

def show_spec(spec, y_axis='mel'):
    librosa.display.specshow(spec, y_axis=y_axis, x_axis='time')

def preemphasis(x):
    from nnmnkwii.preprocessing import preemphasis
    return preemphasis(x, hparams.preemphasis)


def inv_preemphasis(x):
    from nnmnkwii.preprocessing import inv_preemphasis
    return inv_preemphasis(x, hparams.preemphasis)

def spectrogram(y, power):
    global _mel_freqs
    stftS = librosa.stft(y, n_fft=hparams.fft_size, hop_length=hparams.hop_size)
    if hparams.use_preemphasis:
        y = preemphasis(y)
    S = librosa.stft(y, n_fft=hparams.fft_size, hop_length=hparams.hop_size)
    if _mel_freqs is None:
        _mel_freqs = librosa.mel_frequencies(S.shape[0], fmin=hparams.fmin)
    _S = librosa.perceptual_weighting(np.abs(S)**power, _mel_freqs, ref=hparams.ref_level_db)
    return _normalize(_S - hparams.ref_level_db), stftS

def inv_spectrogram(S):
    y = librosa.istft(S, hop_length=hparams.hop_size)
    #if hparams.use_preemphasis:
    #    return inv_preemphasis(y)
    return y
    

# Conversions:
_mel_freqs = None

def _amp_to_db(x):
    min_level = np.exp(hparams.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _db_to_amp(x):
    return np.power(10.0, x * 0.05)


def _normalize(S):
    return np.clip((S - hparams.min_level_db) / -hparams.min_level_db, 0, 1)


def _denormalize(S):
    return (np.clip(S, 0, 1) * -hparams.min_level_db) + hparams.min_level_db
