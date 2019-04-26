class hparams:

    #--------------     
    # audio processing parameters
    fmin = 125
    fft_size = 1024
    stft_frames = 25
    stft_stride = 1
    hop_size = 256
    win_length = 1024
    sample_rate = 22050
    use_preemphasis = True # apply preemphasis transformation to waveform
    preemphasis = 0.97
    mix_power_factor = 2
    vox_power_factor = 3
    min_level_db = -100
    ref_level_db = 20
    rescaling = False
    rescaling_max = 0.999
    allow_clipping_in_normalization = True
    eval_length = sample_rate*2  # slice size for evaluation samples
    #----------------
    #
    #----------------
    # model parameters
    model_type='resnet18'  # convnet or resnet
    init_conv_kernel = (7, 3)
    kernel = (3, 3)
    # convert target spectrogram to mask at this activity threshold
    mask_threshold = 0.5
    # convert output to binary mask at inference time
    mask_at_eval = False
    # threshold for masking at inference time
    eval_mask_threshold = 0.1
    #----------------
    #
    #----------------
    # training parameters
    batch_size = 256
    test_batch_size = 32
    nepochs = 50
    train_loss_every_step = 100
    valid_every_step = 7517
    save_every_step = 7517
    eval_every_epoch = 1
    num_evals = 4  # number of evals to generate
    validation_size = 256000
    grad_norm = 10
    #learning rate parameters
    initial_learning_rate=1e-4
    lr_schedule_type = 'cyclic' # or 'noam'
    # for noam learning rate schedule
    noam_warm_up_steps = 2000 * (batch_size // 16)
    # for step learning rate schedule
    step_gamma = 0.1
    lr_step_interval = 8000
    # for cyclic learning rate schedule
    min_lr = 2e-5
    max_lr = 1e-4
    cycles_per_epoch = 4

    adam_beta1=0.9
    adam_beta2=0.999
    adam_eps=1e-8
    amsgrad=False
    weight_decay = 0.0
    fix_learning_rate = None  # Use one of the learning rate schedules if not None
    #-----------------
