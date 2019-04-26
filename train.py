"""Training Vocal-Mask model

usage: train.py [options] <data-root>

options:
    --checkpoint-dir=<dir>      Directory where to save model checkpoints [default: checkpoints].
    --checkpoint=<path>         Restore model from checkpoint path if given.
    -h, --help                  Show this help message and exit
"""
from docopt import docopt

import os
from os.path import dirname, join, expanduser
import random
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pickle

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from audio import *
from model import build_model
from dataset import basic_collate, SpectrogramDataset
from hparams import hparams as hp
from lrschedule import noam_learning_rate_decay, step_learning_rate_decay
import discordhook

global_step = 0
global_epoch = 0
global_test_step = 0
train_losses = []
valid_losses = []
use_cuda = torch.cuda.is_available()

def save_checkpoint(device, model, optimizer, step, checkpoint_dir, epoch):
    checkpoint_path = join(
        checkpoint_dir, "checkpoint_step{:09d}.pth".format(step))
    optimizer_state = optimizer.state_dict()
    global global_test_step
    global train_losses
    global valid_losses
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
        "global_test_step": global_test_step,
        "train_losses": train_losses,
        "valid_losses":valid_losses
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)


def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def load_checkpoint(path, model, optimizer, reset_optimizer):
    global global_step
    global global_epoch
    global global_test_step
    global train_losses
    global valid_losses

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    model.load_state_dict(checkpoint["state_dict"])
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    global_step = checkpoint["global_step"]
    global_epoch = checkpoint["global_epoch"]
    global_test_step = checkpoint.get("global_test_step", 0)
    train_losses = checkpoint["train_losses"]
    valid_losses = checkpoint["valid_losses"]

    return model


def test_save_checkpoint():
    checkpoint_path = "checkpoints/"
    device = torch.device("cuda" if use_cuda else "cpu")
    model = build_model()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    global global_step, global_epoch, global_test_step
    save_checkpoint(device, model, optimizer, global_step, checkpoint_path, global_epoch)

    model = load_checkpoint(checkpoint_path+"checkpoint_step000000000.pth", model, optimizer, False)


def evaluate_model(device, model, path, checkpoint_dir, global_epoch):
    """evaluate model by generating sample spectrograms

    """

    mix_path = os.path.join(path, "mix")
    vox_path = os.path.join(path, "vox")
    files = os.listdir(mix_path)
    random.shuffle(files)
    print("Evaluating model...")
    for f in tqdm(files[:hp.num_evals]):
        mix_wav = load_wav(os.path.join(mix_path, f))
        vox_wav = load_wav(os.path.join(vox_path,f))
        S = model.generate_specs(device, mix_wav)
        mix_spec = spectrogram(mix_wav, power=hp.mix_power_factor)[0]
        vox_spec = spectrogram(vox_wav, power=hp.vox_power_factor)[0]
        file_id = f.split(".")[0]
        fig_path = os.path.join(checkpoint_dir, 'eval', f'epoch_{global_epoch:06d}_vox_spec_{file_id}.png')
        plt.figure()
        plt.subplot(221)
        plt.title("Mixture")
        show_spec(mix_spec)

        plt.subplot(222)
        plt.title("Ground Truth Vocal")
        show_spec(vox_spec)

        plt.subplot(223)
        plt.title("Generated Mask")
        show_spec(S["mask"]["vocals"])

        plt.subplot(224)
        plt.title("Applied Mask")
        show_spec(S["spec"]["vocals"])

        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close('all')


def validation_step(device, model, testloader, criterion):
    """check loss on validation set
    """

    model.eval()
    print("")
    running_loss = 0
    for i, (x, y) in enumerate(tqdm(testloader)):
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        loss = criterion(y_pred, y)
        running_loss += loss.item()
        avg_loss = running_loss / (i+1)
    return avg_loss


def get_learning_rate(global_step, n_iters):
    if hp.fix_learning_rate:
        current_lr = hp.fix_learning_rate
    elif hp.lr_schedule_type == 'step':
        current_lr = step_learning_rate_decay(hp.initial_learning_rate, 
                    global_step, hp.step_gamma, hp.lr_step_interval)
    elif hp.lr_schedule_type == 'cyclic':
        step_size = n_iters//(2*hp.cycles_per_epoch)
        cycle = np.floor(1 + global_step/(2*step_size))
        x = abs(global_step/step_size - 2*cycle + 1)
        current_lr = hp.min_lr + (hp.max_lr - hp.min_lr)*max(0, (1-x))
    else:
        current_lr = noam_learning_rate_decay(hp.initial_learning_rate, 
                    global_step, hp.noam_warm_up_steps)
    return current_lr


def train_loop(device, model, trainloader, testloader,  optimizer, checkpoint_dir, eval_dir):
    """Main training loop.

    """
    #criterion = torch.nn.MSELoss()
    criterion = torch.nn.BCELoss()

    global global_step, global_epoch, global_test_step, train_losses, valid_losses
    n_iters = int(np.ceil(len(trainloader.dataset)/hp.batch_size))
    while global_epoch < hp.nepochs:
        running_loss = 0
        model.train()
        print(f"[Epoch {global_epoch}]")
        for i, (x, y) in enumerate(tqdm(trainloader)):
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)

            # calculate learning rate and update learning rate
            current_lr = get_learning_rate(global_step, n_iters)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            optimizer.zero_grad()
            loss.backward()

            # clip gradient norm
            nn.utils.clip_grad_norm_(model.parameters(), hp.grad_norm)
            optimizer.step()

            running_loss += loss.item()
            avg_loss = running_loss / (i+1)
            global_step += 1
            if global_step % hp.train_loss_every_step == 0:
                train_losses.append((global_step, loss.item()))

            if global_step % 1000 == 0:
                discordhook.send_message(f"Step:{global_step}, lr:{current_lr}, training loss:{loss.item():.6f}")
            # save checkpoint
            if global_step != 0 and global_step % hp.save_every_step == 0:
                save_checkpoint(device, model, optimizer, global_step, checkpoint_dir, global_epoch)

            # Validation
            if global_step != 0 and global_step % hp.valid_every_step == 0:
                avg_valid_loss = validation_step(device, model, testloader, criterion)
                valid_losses.append((global_step, avg_valid_loss))
                msg = (f"Step:{global_step}, lr:{current_lr}, training loss:{avg_loss:.6f}, validation loss:{avg_valid_loss:.6f}")
                print(msg)
                discordhook.send_message(msg)

        # save checkpoint
        save_checkpoint(device, model, optimizer, global_step, checkpoint_dir, global_epoch)

        # Evaluation
        if global_epoch % hp.eval_every_epoch == 0:
            evaluate_model(device, model, eval_dir, checkpoint_dir, global_epoch)
    
        global_epoch += 1


if __name__=="__main__":
    args = docopt(__doc__)
    #print("Command line args:\n", args)
    checkpoint_dir = args["--checkpoint-dir"]
    checkpoint_path = args["--checkpoint"]
    data_root = args["<data-root>"]
    eval_dir = os.path.join(data_root, "eval")

    # make dirs, load dataloader and set up device
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.join(checkpoint_dir,'eval'), exist_ok=True)
    device = torch.device("cuda" if use_cuda else "cpu")
    print("using device:{}".format(device))


    # build model, create optimizer
    model = build_model().to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=hp.initial_learning_rate, betas=(
        hp.adam_beta1, hp.adam_beta2),
        eps=hp.adam_eps, weight_decay=hp.weight_decay,
        amsgrad=hp.amsgrad)

    if hp.fix_learning_rate:
        print("using fixed learning rate of :{}".format(hp.fix_learning_rate))
    elif hp.lr_schedule_type == 'step':
        print("using exponential learning rate decay")
    elif hp.lr_schedule_type == 'noam':
        print("using noam learning rate decay")
    elif hp.lr_schedule_type == 'cyclic':
        print("using cyclic learning rate")

    # load checkpoint
    if checkpoint_path is None:
        print("no checkpoint specified as --checkpoint argument, creating new model...")
        
    else:
        model = load_checkpoint(checkpoint_path, model, optimizer, False)
        print("loading model from checkpoint:{}".format(checkpoint_path))
        # set global_test_step to True so we don't evaluate right when we load in the model
        global_test_step = True

    # create dataloaders
    with open(os.path.join(data_root, 'spec_info.pkl'), 'rb') as f:
        spec_info = pickle.load(f)
    test_path = os.path.join(data_root, "test")
    with open(os.path.join(test_path, "test_spec_info.pkl"), 'rb') as f:
        test_spec_info = pickle.load(f)
    test_specs = test_spec_info
    train_specs = spec_info
    trainset = SpectrogramDataset(data_root, train_specs)
    testset = SpectrogramDataset(test_path, test_specs)
    random.shuffle(testset.metadata)
    testset.metadata = testset.metadata[:hp.validation_size]
    print(f"# Training examples: {len(trainset)}")
    print(f"# Validation examples: {len(testset)}")
    trainloader = DataLoader(trainset, collate_fn=basic_collate, shuffle=True, num_workers=0, batch_size=hp.batch_size)
    testloader = DataLoader(testset, collate_fn=basic_collate, shuffle=True, num_workers=0, batch_size=hp.test_batch_size)


    # main train loop
    try:
        train_loop(device, model, trainloader, testloader, optimizer, checkpoint_dir, eval_dir)
    except KeyboardInterrupt:
        print("Interrupted!")
        pass
    finally:
        print("saving model....")
        save_checkpoint(device, model, optimizer, global_step, checkpoint_dir, global_epoch)
    

