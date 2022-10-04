"""
Fingers flexion cropped decoding on BCIC IV 4 ECoG Dataset
==========================================================

This tutorial shows you how to train and test deep learning models with
Braindecode on ECoG BCI IV competition dataset 4 using cropped mode. For this
dataset we will predict 5 regression targets corresponding to flexion of each finger.
The targets were recorded as a time series (each 25 Hz), so this tutorial is
an example of time series target prediction.
"""

import copy
import pandas as pd
import numpy as np
import sklearn
from mne import set_log_level
from braindecode.datasets import BCICompetitionIVDataset4

accuracy_list = []
for i in range(3):
    subject_id = i + 1
    dataset = BCICompetitionIVDataset4(subject_ids=[subject_id])

    ######################################################################
    # Split dataset into train and test
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #
    # We can easily split the dataset using additional info stored in the
    # description attribute, in this case ``session`` column. We select `train` dataset
    # for training and validation and `test` for final evaluation.
    dataset = dataset.split('session')
    train_set = dataset['train']
    test_set = dataset['test']

    ######################################################################
    # Preprocessing
    # ~~~~~~~~~~~~~
    #
    # Now we apply preprocessing like bandpass filtering to our dataset. You
    # can either apply functions provided by
    # `mne.Raw <https://mne.tools/stable/generated/mne.io.Raw.html>`__ or
    # `mne.Epochs <https://mne.tools/0.11/generated/mne.Epochs.html#mne.Epochs>`__
    # or apply your own functions, either to the MNE object or the underlying
    # numpy array.
    #
    # .. note::
    #    Preprocessing steps are taken from a standard EEG processing pipeline.
    #    The only change is the cutoff frequency of the filter. For a proper ECoG
    #    decoding other preprocessing steps may be needed.
    #
    # .. note::
    #    These prepocessings are now directly applied to the loaded
    #    data, and not on-the-fly applied as transformations in
    #    PyTorch-libraries like
    #    `torchvision <https://pytorch.org/docs/stable/torchvision/index.html>`__.
    #


    from braindecode.preprocessing import (
        exponential_moving_standardize, preprocess, Preprocessor)

    low_cut_hz = 1.  # low cut frequency for filtering
    high_cut_hz = 200.  # high cut frequency for filtering, for ECoG higher than for EEG
    # Parameters for exponential moving standardization
    factor_new = 1e-3
    init_block_size = 1000

    ######################################################################
    # We select only first 30 seconds from the training dataset to limit time and memory
    # to run this example. We split training dataset into train and validation (only 6 seconds).
    # To obtain full results whole datasets should be used.

    # valid_set = preprocess(copy.deepcopy(train_set), [Preprocessor('crop', tmin=24, tmax=30)])
    # preprocess(train_set, [Preprocessor('crop', tmin=0, tmax=24)])
    # preprocess(test_set, [Preprocessor('crop', tmin=0, tmax=24)])

    valid_set = preprocess(copy.deepcopy(train_set), [Preprocessor('crop')])
    preprocess(train_set, [Preprocessor('crop')])
    preprocess(test_set, [Preprocessor('crop')])


    ######################################################################
    # In time series targets setup, targets variables are stored in mne.Raw object as channels
    # of type `misc`. Thus those channels have to be selected for further processing. However,
    # many mne functions ignore `misc` channels and perform operations only on data channels
    # (see https://mne.tools/stable/glossary.html#term-data-channels).
    preprocessors = [
        # TODO: ensure that misc is not removed
        Preprocessor('pick_types', ecog=True, misc=True),
        Preprocessor(lambda x: x / 1e6, picks='ecog'),  # Convert from V to uV
        Preprocessor('filter', l_freq=low_cut_hz, h_freq=high_cut_hz),  # Bandpass filter
        Preprocessor(exponential_moving_standardize,  # Exponential moving standardization
                    factor_new=factor_new, init_block_size=init_block_size, picks='ecog')
    ]
    # Transform the data
    preprocess(train_set, preprocessors)
    preprocess(valid_set, preprocessors)
    preprocess(test_set, preprocessors)

    # Extract sampling frequency, check that they are same in all datasets
    sfreq = train_set.datasets[0].raw.info['sfreq']
    assert all([ds.raw.info['sfreq'] == sfreq for ds in train_set.datasets])
    # Extract target sampling frequency
    target_sfreq = train_set.datasets[0].raw.info['temp']['target_sfreq']


    ######################################################################
    # Create model
    # ------------
    #
    # In contrast to trialwise decoding, we first have to create the model
    # before we can cut the dataset into windows. This is because we need to
    # know the receptive field of the network to know how large the window
    # stride should be.
    #
    # We first choose the compute/input window size that will be fed to the
    # network during training This has to be larger than the networks
    # receptive field size and can otherwise be chosen for computational
    # efficiency (see explanations in the beginning of this tutorial). Here we
    # choose 1000 samples, which is 1 second for the 1000 Hz sampling rate.
    #

    input_window_samples = 1000


    ######################################################################
    # Now we create the deep learning model! Braindecode comes with some
    # predefined convolutional neural network architectures for raw
    # time-domain EEG. Here, we use the shallow ConvNet model from `Deep
    # learning with convolutional neural networks for EEG decoding and
    # visualization <https://arxiv.org/abs/1703.05051>`__. These models are
    # pure `PyTorch <https://pytorch.org>`__ deep learning models, therefore
    # to use your own model, it just has to be a normal PyTorch
    # `nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__.
    #

    import torch
    from braindecode.util import set_random_seeds
    from braindecode.models import ShallowFBCSPNet

    cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
    # cuda = False
    device = 'cuda' if cuda else 'cpu'
    if cuda:
        torch.backends.cudnn.benchmark = True
    # Set random seed to be able to roughly reproduce results
    # Note that with cudnn benchmark set to True, GPU indeterminism
    # may still make results substantially different between runs.
    # To obtain more consistent results at the cost of increased computation time,
    # you can set `cudnn_benchmark=False` in `set_random_seeds`
    # or remove `torch.backends.cudnn.benchmark = True`
    seed = 20200220
    set_random_seeds(seed=seed, cuda=cuda)

    n_classes = 1
    # Extract number of chans and time steps from dataset
    n_chans = train_set[0][0].shape[0] - 5

    model = ShallowFBCSPNet(
        n_chans,
        n_classes,
        final_conv_length=2,
    )
    # We are removing the softmax layer to make it a regression model
    new_model = torch.nn.Sequential()
    for name, module_ in model.named_children():
        if "softmax" in name:
            continue
        new_model.add_module(name, module_)
    model = new_model

    # Send model to GPU
    if cuda:
        model.cuda()

    from braindecode.models import to_dense_prediction_model, get_output_shape

    to_dense_prediction_model(model)


    ######################################################################
    # To know the models’ receptive field, we calculate the shape of model
    # output for a dummy input.

    n_preds_per_input = get_output_shape(model, n_chans, input_window_samples)[2]


    ######################################################################
    # Cut Compute Windows
    # ~~~~~~~~~~~~~~~~~~~
    #

    from braindecode.preprocessing import create_fixed_length_windows

    # Create windows using braindecode function for this. It needs parameters to define how
    # trials should be used.

    train_set = create_fixed_length_windows(
        train_set,
        start_offset_samples=0,
        stop_offset_samples=None,
        window_size_samples=input_window_samples,
        window_stride_samples=n_preds_per_input,
        drop_last_window=False,
        targets_from='channels',
        last_target_only=False,
        preload=False
    )

    valid_set = create_fixed_length_windows(
        valid_set,
        start_offset_samples=0,
        stop_offset_samples=None,
        window_size_samples=input_window_samples,
        window_stride_samples=n_preds_per_input,
        drop_last_window=False,
        targets_from='channels',
        last_target_only=False,
        preload=False
    )

    test_set = create_fixed_length_windows(
        test_set,
        start_offset_samples=0,
        stop_offset_samples=None,
        window_size_samples=input_window_samples,
        window_stride_samples=n_preds_per_input,
        drop_last_window=False,
        targets_from='channels',
        last_target_only=False,
        preload=False
    )
    ######################################################################
    # We select only the thumb's finger flexion to create one model per finger.
    #
    # .. note::
    #    Methods to predict all 5 fingers flexion with the same model may be considered as well.
    #    We encourage you to find your own way to use braindecode models to predict fingers flexions.
    #
    train_set.target_transform = lambda x: x[0: 1]
    valid_set.target_transform = lambda x: x[0: 1]
    test_set.target_transform = lambda x: x[0: 1]

    #####################################################################
    # Training
    # --------
    #
    # In difference to trialwise decoding, we now should supply
    # ``cropped=True`` to the EEGClassifier, and ``CroppedLoss`` as the
    # criterion, as well as ``criterion__loss_function`` as the loss function
    # applied to the meaned predictions.
    #
    # .. note::
    #    In this tutorial, we use some default parameters that we
    #    have found to work well for EEG motor decoding, however we strongly
    #    encourage you to perform your own hyperparameter optimization using
    #    cross validation on your training data.
    #

    from skorch.callbacks import LRScheduler
    from skorch.helper import predefined_split

    from braindecode.training import TimeSeriesLoss
    from braindecode import EEGRegressor
    from braindecode.training import CroppedTimeSeriesEpochScoring

    # These values we found good for shallow network for EEG MI decoding:
    lr = 0.0625 * 0.01
    weight_decay = 0
    batch_size = 64
    n_epochs = 130

    regressor = EEGRegressor(
        model,
        cropped=True,
        aggregate_predictions=False,
        criterion=TimeSeriesLoss,
        criterion__loss_function=torch.nn.functional.mse_loss,
        optimizer=torch.optim.AdamW,
        train_split=predefined_split(valid_set),
        optimizer__lr=lr,
        optimizer__weight_decay=weight_decay,
        iterator_train__shuffle=True,
        batch_size=batch_size,
        callbacks=[
            ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
            ('r2_train', CroppedTimeSeriesEpochScoring(sklearn.metrics.r2_score,
                                                    lower_is_better=False,
                                                    on_train=True,
                                                    name='r2_train')
            ),
            ('r2_valid', CroppedTimeSeriesEpochScoring(sklearn.metrics.r2_score,
                                                    lower_is_better=False,
                                                    on_train=False,
                                                    name='r2_valid')
            )
        ],
        device=device,
    )
    set_log_level(verbose='WARNING')

    ######################################################################
    # Model training for a specified number of epochs. ``y`` is None as it is already supplied
    # in the dataset.
    regressor.fit(train_set, y=None, epochs=n_epochs)

    results_columns = ['r2_train', 'r2_valid', 'train_loss', 'valid_loss']
    score_df = pd.DataFrame(regressor.history[:, results_columns], columns=results_columns,
                            index=regressor.history[:, 'epoch'])
    accuracy_list.append(score_df.r2_valid[n_epochs])

print('Average r2 valid list: {}'.format(accuracy_list))
print('Average r2 valid score: {}'.format(np.array(accuracy_list).mean()))