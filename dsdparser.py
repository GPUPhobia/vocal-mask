import os
import os.path
import librosa
from hparams import hparams as hp

class Stem(object):
    def __init__(self, path, targets=None):
        self.audio = path
        if targets:
            self.targets = targets

    def load(self):
        return librosa.load(self.audio, sr=hp.sample_rate)[0]

class DSD(object):
    def __init__(self, root_dir, mix_dir="Mixtures", src_dir="Sources"):
        self.root_dir = root_dir
        self.mixtures = os.path.join(root_dir, mix_dir)
        self.sources = os.path.join(root_dir, src_dir)

    def _parse(self):
        tracks = []
        mixfs = os.listdir(self.mixtures)
        srcfs = os.listdir(self.sources)
        total = len(mixfs) + 1
        for i in range(1,total):
            mixdir = next(f for f in mixfs if int(f[:3]) == i)
            srcdir = next(f for f in srcfs if int(f[:3]) == i)
            targets = {}
            mixpath = os.path.join(self.mixtures, mixdir)
            srcpath = os.path.join(self.sources, srcdir)
            mix = os.listdir(mixpath)[0]
            srcs = os.listdir(srcpath)
            for src in srcs:
                fname = src[:-4]
                stem = Stem(os.path.join(srcpath, src))
                targets[fname] = stem
            tracks.append(Stem(os.path.join(mixpath, mix), targets=targets))
        return tracks

    def load_tracks(self):
        return self._parse()
