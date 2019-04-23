# Vocal Mask CNN
<head><script src="http://api.html5media.info/1.1.8/html5media.min.js"></script></head>

Audio source separation in the music domain.

## Overview

### Problem Formulation

Given a piece of music as input, we want to separate the vocals from the accompanying instrumentation. There are many use cases for this - in music production, a producer may want to extract the vocals from a sample for creative purposes, or a VOIP (voice over IP) application may use it to enhance audio clarity. 

### Approach

Since generative models were difficult to train and the model potentially very large, especially in audio space, we decided it would be best to work with images by converting the input audio to spectrograms via **Short Time Fourier Transforms** (STFT). This has a lot of precedent in previous audio-related models as spectrograms can encapsulate information about temporally-related frequences in a spatially-related fashion. 

How should we extract the vocals based on the spectrogram? We wanted to reduce the problem to a simpler problem as the generative models we tried first were difficult to train to produce something other than white noise. What we came to use is to convert the vocal spectrogram to a binary mask, where 1s represent the presence of vocal activity and 0s represent the lack of vocal activity. The model should learn to do a pixel-level classification of the input spectrogram as either containing or not containing vocal activity. At inference time, the mask can be applied to the mixture spectrogram to generate the vocal-only spectrogram. 

Due to exploration in other works in the audio domain, we initially chose to use **Mel-scale Spectrograms** as our spectrogram of choice. Mel spectrograms use a transformation matrix to convert a STFT-generated spectrogram to a Mel-scaled spectrogram, which attempts to weight magnitudes of frequencies according to their approximate sensitivity to the human ear. This is useful for speech related models as the highest sensitivity is in the human vocal range. 

However, the biggest issue with this approach is that recovering the audio from a Mel spectrogram is not trivial. The Mel-transform matrix is non-square, so performing an inverse matrix multiplication to recover the STFT spectrogram is not really possible. One solution we found was to use a [WaveRNN model](https://github.com/G-Wang/WaveRNN-Pytorch) to recover the audio, since this project uses Mel spectrograms as input. We found it very challenging to train this model on musical datasets, and even with speech datasets the audio may not be properly recovered as the Mel spectrogram has a very limited number of frequency bins, thus significant information about the original pitch of the audio is lost.  

The solution to this was fairly obvious in hindsight, which is to simply use the original STFT spectrogram instead of the Mel spectrogram. The audio can be recovered via inverse STFT, and although some information is lost, the results are quite good qualitatively. We still wanted to keep the Mel weightings, so what we did instead was to create spectrograms via STFT, and apply [Mel perceptual weighting](https://librosa.github.io/librosa/generated/librosa.core.perceptual_weighting.html). One big advantage of using the full STFT with the masking approach is that we are able to keep the phasing information when masking the mixture, which is necessary for recovering high quality audio.

*TODO: Add diagrams and sample waveforms here to demonstrate that the masking approach has potential*

<p align="center">
    <img src="assets/model_approach.png"/>
</p>

This is the approach we ended up with. Input mixture waveforms and target vocal waveforms would be sliced up and converted to spectrograms, then Mel perceptual weightings would be applied. The magnitude of each frequency was also cubed in order to enhance the separation between stronger and softer signals, then the output was scaled to the 0-1 range. 

For the vocal spectrogram, only the center column of the image is kept. This is converted to a binary mask and used as the target label with size (513,). The mixture spectrograms pass through the convolutional neural network, which ends with a 513-way fully-connected layer with a sigmoid to constrain the output to the 0-1 range. We chose to use Binary Cross Entropy Loss as it was well suited for our output and target. 

For the convolutional neural network, we initially tried a fairly basic CNN with 4 convolution layers with ReLU activation and pooling layers, followed by 2 fully-connected layers. However, we were able to achieve a better validation loss with a small residual network.

<p align="center">
    <img src="assets/model_inference.png"/>
</p>

At inference time, the input waveforms are sliced into overlapping windows. Each window is converted to Mel-weighted spectrogram and passed through the network to generate the binary mask. The masks are then concatenated and applied to the pre-Mel-weighted spectrogram (which preserves magnitude and phasing information) to produce the isolated vocal-only spectrogram. To produce the background-only spectrogram, the mask can be inverted. The audio can then be recovered via inverse STFT.

### Past Works

*TODO*

## Usage

### Dataset

We used the [DSD100 dataset](https://sigsep.github.io/datasets/dsd100.html) for this project. The `build_dataset.py` script converts the input audio to spectrogram and saves index slices of the spectrogram. At the smallest slice striding, 100 songs produces approximately 2 million example slices. Not only does this produce more examples for training, but we found the overlapping windows also served the purpose of data-augmentation as it is essentially a horizontal-translation augmentation. We quickly noticed that more examples was improving our models learning ability much more than the complexity and number of parameters of our model, so we augmented the original DSD100 dataset by manually assembling mixture and vocal examples. The process for this was to find waveform stems from remix competitions, sum the stems to produce the mixture, then sum the vocal stems to produce the vocal waveform. 

To generate the dataset, download the DSD100 dataset, then move all files in the Mixtures/Dev and Mixtures/Test directories to the Mixtures/ directory, and do the same for Sources/Dev and Sources/Test. Then, delete the Dev and Test subdirectories and run the script as follows:  
 
```python build_dataset.py <DSD100 root dir> <output dir>```

The window size and striding for the slices are controlled by `hparams.stft_frames` and `hparams.stft_stride`, respectively.

### Training

```python train.py <data dir> --checkpoint=<path to checkpoint file (*.pth)>```

The first argument should be the same as the output directory of `build_dataset`.  
A pretrained model for `hparams.model_type = 'resnet18'` can be downloaded here: [resnet18_step000033000.pth](https://drive.google.com/open?id=19QciqI26LXrJtQqPiilPzhQHNGbx00pP).

*TODO*

### Testing

```python generate.py <path to checkpoint file (*.pth)> <path to mixture wav>```  

This will generate a vocal wav file in the `generated` directory. Below are the parameters in `hparams.py` that control how the mask is applied during inference. This can have some differences in the way the output audio sounds.  
- `hparams.mask_at_eval` - If `True`, the model output will be converted to a mask. If `False`, it will be left as values in the range (0,1). This can produce less harsh cutoffs.  
- `hparams.eval_mask_threshold` - If `mask_at_eval` is `True`, use this to set the masking threshold. Range (0,1).  
- `hparams.noise_gate` - If `mask_at_eval` is `False`, use this to set a cutoff where all values below this threshold are zeroed. Range (0,1).

## Results

*TODO*

### Training

### Example Audio  
Mixture  
<audio src="assets/audio_samples/mixture/1.wav" controls preload></audio>

## Discussion

## References

[1] End-to-end music source separation: is it possible in the waveform domain? https://arxiv.org/abs/1810.12187  
[2] Monoaural Audio Source Separation Using Deep Convolutional Neural Networks. http://mtg.upf.edu/node/3680  
[3] Efficient Neural Audio Synthesis. https://arxiv.org/abs/1802.08435v1  
[4] Liutkus, A., Stoter, F.R., Rafii, Z. The 2016 Signal Separation Evaluation Campaign. https://sigsep.github.io/datasets/dsd100.html  

[5] Cyclic Learning Rates. https://arxiv.org/abs/1506.01186  
