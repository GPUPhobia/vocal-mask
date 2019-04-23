"""Plot loss generated from training.

usage: plotloss.py <checkpoint-path>

options:
    -h, --help                  Show this help message and exit
"""
import torch
from docopt import docopt
import platform

import matplotlib
if platform.system() == 'Darwin':
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def _load(checkpoint_path):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def load_checkpoint(path):
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    train_losses = checkpoint["train_losses"]
    valid_losses = checkpoint["valid_losses"]
    return train_losses, valid_losses

def plot_loss(train_losses, valid_losses):
    print(sorted(valid_losses, key=lambda x: x[1]))
    plt.figure()
    plt.title("Binary Cross Entropy Loss")
    trainX, trainY = zip(*train_losses)
    validX, validY = zip(*valid_losses)
    plt.plot(trainX, trainY, label="Training")
    plt.plot(validX, validY, label="Validation")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

args = docopt(__doc__)
checkpoint_path = args["<checkpoint-path>"]
train_losses, valid_losses = load_checkpoint(checkpoint_path)
plot_loss(train_losses, valid_losses)
