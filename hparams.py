class hparams:

    #--------------     
    # audio processing parameters
    num_mels = 280
    fmin = 125
    fmax = 11025
    fft_size = 1024
    stft_frames = 25
    stft_stride = 1
    hop_size = 256
    win_length = 1024
    sample_rate = 22050
    use_preemphasis = True # apply preemphasis transformation to waveform
    preemphasis = 0.97
    power = 3
    min_level_db = -100
    ref_level_db = 20
    lws_mode = 'speech' # speech or music
    rescaling = False
    rescaling_max = 0.999
    allow_clipping_in_normalization = True
    trim = False # whether to cut silence from the ends of the waveform
    trim_thresh = 80 # how much below max db to trim waveform
    eval_length = sample_rate*2  # slice size for evaluation samples
    #----------------
    #
    #----------------
    # model parameters
    model_type='resnet'  # convnet or resnet
    res_dims = [
        (32, 32, None), (32, 32, (2,2)),
        (32, 64, None), (64, 64, None), (64, 16, (2,2)) 
    ]
    # convert target spectrogram to mask at this activity threshold
    mask_threshold = 0.5
    # convert output to binary mask at inference time
    mask_at_eval = True
    # if not mask_at_eval is False, zero out values under the noise_gate
    noise_gate = 0.3
    #----------------
    #
    #----------------
    # training parameters
    batch_size = 96
    test_batch_size = 8
    nepochs = 100
    save_every_epoch = 2
    eval_every_epoch = 2
    num_evals = 4  # number of evals to generate
    train_test_split = 0.05 # reserve 5% of data for validation
    # seq_len_factor can be adjusted to increase training sequence length (will increase GPU usage)
    seq_len_factor = 5
    seq_len = seq_len_factor * hop_size
    grad_norm = 10
    #learning rate parameters
    initial_learning_rate=1e-4
    lr_schedule_type = 'step' # or 'noam'
    # for noam learning rate schedule
    noam_warm_up_steps = 2000 * (batch_size // 16)
    # for step learning rate schedule
    step_gamma = 0.1
    lr_step_interval = 30000

    adam_beta1=0.9
    adam_beta2=0.999
    adam_eps=1e-8
    amsgrad=False
    weight_decay = 0.0
    #fix_learning_rate = 5e-6 # modify if one wants to use a fixed learning rate, else set to None to use noam learning rate
    fix_learning_rate = 1e-4
    #-----------------
