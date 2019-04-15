import librosa
import librosa.effects
import librosa.feature
import librosa.filters
import math
import numpy as np
from scipy import signal
from hparams import hparams
from scipy.io import wavfile

# r9r9 preprocessing
import lws


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

def spectrogram(y):
    global _mel_freqs
    if hparams.use_preemphasis:
        y = preemphasis(y)
    S = librosa.stft(y, n_fft=hparams.fft_size, hop_length=hparams.hop_size)
    if _mel_freqs is None:
        _mel_freqs = librosa.mel_frequencies(S.shape[0], fmin=hparams.fmin)
    _S = librosa.perceptual_weighting(np.abs(S)**2, _mel_freqs, ref=hparams.ref_level_db)
    return _normalize(_S - hparams.ref_level_db), S

def inv_spectrogram(S):
    y = librosa.istft(S, hop_length=hparams.hop_size)
    #y = griffinlim(S, n_iter=n_iter, hop_length=hparams.hop_size)
    if hparams.use_preemphasis:
        return inv_preemphasis(y)
    return y
    

def melspectrogram(y):
    if hparams.use_preemphasis:
        y = preemphasis(y)
    S = librosa.feature.melspectrogram(
            y, sr=hparams.sample_rate, n_fft=hparams.fft_size, 
            hop_length=hparams.hop_size, power=2.0, n_mels=hparams.num_mels)
    return _normalize(librosa.power_to_db(S, ref=hparams.ref_level_db) - hparams.ref_level_db)

def inv_melspectrogram(spectrogram, n_iter=32):
    '''Converts melspectrogram to waveform using librosa'''

    S = _denormalize(spectrogram) + hparams.ref_level_db
    S = _mel_to_linear(librosa.db_to_power(S, ref=hparams.ref_level_db), power=2.0)
    y = griffinlim(S, n_iter=n_iter, hop_length=hparams.hop_size, 
            win_length=None, window='hann', center=True, dtype=np.float32,
            length=None, pad_mode='reflect')
    if hparams.use_preemphasis:
        return inv_preemphasis(y)
    return y

def griffinlim(S, n_iter=32, hop_length=None, win_length=None, window='hann',
                center=True, dtype=np.float32, length=None, pad_mode='reflect'):
    n_fft = 2*(S.shape[0] - 1)
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    for _ in range(n_iter):
        inverse = librosa.istft(S*angles, hop_length=hop_length, win_length=win_length,
            window=window, center=center, dtype=dtype, length=length)
        rebuilt = librosa.stft(inverse, n_fft=n_fft, hop_length=hop_length,
                    win_length=win_length, window=window, center=center, pad_mode=pad_mode)
        angles[:] = np.exp(1j * np.angle(rebuilt))
    return librosa.istft(S*angles, hop_length=hop_length, win_length=win_length,
            window=window, center=center, dtype=dtype, length=length)


def _lws_processor():
    return lws.lws(hparams.fft_size, hparams.hop_size, mode=hparams.lws_mode)


# Conversions:


_mel_basis = None
_mel_freqs = None

def _linear_to_mel(spectrogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectrogram)

def _mel_to_linear(spectrogram, power):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.clip(np.linalg.lstsq(_mel_basis, spectrogram, rcond=None)[0], 0, None)**(1./power)


def _build_mel_basis():
    if hparams.fmax is not None:
        assert hparams.fmax <= hparams.sample_rate // 2
    return librosa.filters.mel(hparams.sample_rate, hparams.fft_size,
                               fmin=hparams.fmin, fmax=hparams.fmax,
                               n_mels=hparams.num_mels)


def _amp_to_db(x):
    min_level = np.exp(hparams.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _db_to_amp(x):
    return np.power(10.0, x * 0.05)


def _normalize(S):
    return np.clip((S - hparams.min_level_db) / -hparams.min_level_db, 0, 1)


def _denormalize(S):
    return (np.clip(S, 0, 1) * -hparams.min_level_db) + hparams.min_level_db


# Fatcord's preprocessing
def quantize(x):
    """quantize audio signal

    """
    quant = (x + 1.) * (2**hparams.bits - 1) / 2
    return quant.astype(np.int)


# testing
def test_everything():
    wav = np.random.randn(12000,)
    mel = melspectrogram(wav)
    spec = spectrogram(wav)
    quant = quantize(wav)
    print(wav.shape, mel.shape, spec.shape, quant.shape)
    print(quant.max(), quant.min(), mel.max(), mel.min(), spec.max(), spec.min())
