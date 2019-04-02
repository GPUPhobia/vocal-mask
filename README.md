# Vocal Mask CNN

Separate vocals from mixture. 

## General Flow
1. Downsample waveforms to 22050Hz
2. Slice mixture and vocal waveforms into ~290ms frames, with a frame stride of ~11.6ms (derived from hop size (256) / sample rate (22050))
3. Convert each frame to melspectrogram (256x25)
4. Select the middle frame of the vocal spectrogram as the target
5. Goal is to predict the middle frame of the vocal spectrogram given the mixture spectrogram (with the context of 12 frames before and after the middle frame)

## Dataset
The DSD100 dataset is used for this project. With 100 songs, sliced into ~290ms windows with a stride of ~11.6ms, and each song averaging about 4 minutes, this gives about 2 million samples for the dataset.
