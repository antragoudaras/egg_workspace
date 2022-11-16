from skorch.callbacks import LRScheduler
from torch import nn
from torch.nn import init
import mne
from mne.io import concatenate_raws
from braindecode.util import set_random_seeds
from braindecode.models import ShallowFBCSPNet
from braindecode.models.util import to_dense_prediction_model, get_output_shape
import numpy as np
import torch
from mne.io import concatenate_raws
from skorch.helper import predefined_split
from torch import optim
from braindecode.datasets.xy import create_from_X_y
from braindecode.training.losses import CroppedLoss
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import create_windows_from_events
from braindecode.preprocessing import (
		exponential_moving_standardize, preprocess, Preprocessor, scale)
from skorch.helper import predefined_split
from braindecode import EEGClassifier
from braindecode.training import CroppedLoss
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import time
import random
import argparse

parser = argparse.ArgumentParser("Generating Low Freq parser")
parser.add_argument("--load-dataset", type=str, default='./random_dataset_optimized_low_freq.xlsx', help="The path where the generated dataset will be stored")
args = parser.parse_args()



df = pd.read_excel(args.load_dataset, index_col=0)
num_of_archs = df.__len__()
accuracy = [0.0 for _ in range(num_of_archs)]
df['accuracy'] = accuracy
num_of_subjects= 9

def Average(lst):
	return sum(lst) / len(lst)

for j in range(num_of_archs):
	avgList = []
	for i in range(num_of_subjects):
		subject_id = i+1
		dataset = MOABBDataset(dataset_name="BNCI2014001", subject_ids=[subject_id])

		low_cut_hz = 0.  # low cut frequency for filtering
		high_cut_hz = 38.  # high cut frequency for filtering
		# Parameters for exponential moving standardization
		factor_new = 1e-3
		init_block_size = 1000

		preprocessors = [
			Preprocessor('pick_types', eeg=True, meg=False, stim=False),  # Keep EEG sensors
			Preprocessor(scale, factor=1e6, apply_on_array=True),  # Convert from V to uV
			Preprocessor('filter', l_freq=low_cut_hz, h_freq=high_cut_hz),  # Bandpass filter
			Preprocessor(exponential_moving_standardize,  # Exponential moving standardization
						 factor_new=factor_new, init_block_size=init_block_size)
		]

		# Transform the data
		preprocess(dataset, preprocessors)

		input_window_samples = 1000


		cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
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

		n_classes = 4
		# Extract number of chans from dataset
		n_chans = dataset[0][0].shape[0]

		model = ShallowFBCSPNet(
			n_chans,
			n_classes,
			input_window_samples=input_window_samples,
			final_conv_length=30,
			n_filters_time=int(df.iloc[j]['n_filters_time']),
			filter_time_length=int(df.iloc[j]['filter_time_length']),
			n_filters_spat=int(df.iloc[j]['n_filters_spat']),
			pool_time_length=int(df.iloc[j]['pool_time_length']),
			pool_time_stride=int(df.iloc[j]['pool_time_stride']),
			drop_prob=df.iloc[j]['drop_prob']
		)

		# Send model to GPU
		if cuda:
			model.cuda()

		to_dense_prediction_model(model)

		n_preds_per_input = get_output_shape(model, n_chans, input_window_samples)[2]

		trial_start_offset_seconds = -0.5
		# Extract sampling frequency, check that they are same in all datasets
		sfreq = dataset.datasets[0].raw.info['sfreq']
		assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])

		# Calculate the trial start offset in samples.
		trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)

		# Create windows using braindecode function for this. It needs parameters to define how
		# trials should be used.
		windows_dataset = create_windows_from_events(
			dataset,
			trial_start_offset_samples=trial_start_offset_samples,
			trial_stop_offset_samples=0,
			window_size_samples=input_window_samples,
			window_stride_samples=n_preds_per_input,
			drop_last_window=False,
			preload=True
		)

		splitted = windows_dataset.split('session')
		train_set = splitted['session_T']
		valid_set = splitted['session_E']

		# These values we found good for shallow network:
		lr = df.iloc[j]['param_7']
		weight_decay = df.iloc[j]['param_8']

		# For deep4 they should be:
		# lr = 1 * 0.01
		# weight_decay = 0.5 * 0.001

		batch_size = 64
		n_epochs = 70

		clf = EEGClassifier(
			model,
			cropped=True,
			criterion=CroppedLoss,
			criterion__loss_function=torch.nn.functional.nll_loss,
			optimizer=torch.optim.AdamW,
			train_split=predefined_split(valid_set),
			optimizer__lr=lr,
			optimizer__weight_decay=weight_decay,
			iterator_train__shuffle=True,
			batch_size=batch_size,
			callbacks=[
				"accuracy", ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
			],
			device=device,
		)
		# Model training for a specified number of epochs. `y` is None as it is already supplied
		# in the dataset.
		print("ARCHITECTURE ",j+1, "SUBJECT ", subject_id)
		clf.fit(train_set, y=None, epochs=n_epochs)

		# Extract loss and accuracy values for plotting from history object
		results_columns = ['train_loss', 'valid_loss', 'train_accuracy', 'valid_accuracy']
		df_scores = pd.DataFrame(clf.history[:, results_columns], columns=results_columns,
						  index=clf.history[:, 'epoch'])

		avgList.append(100*df_scores.valid_accuracy[n_epochs])

	average = Average(avgList)
	print("Arch.{} average valid accuarcy across all subjects:{}".format(j+1,average))
	df.at[j, 'accuracy'] = average

prefix = None
if "random_dataset_" in args.load_dataset:
	prefix = "random_set"
elif "train_dataset_" in args.load_dataset:
	prefix = "train_set"
elif "val_dataset_" in args.load_dataset:
	prefix = "val_set"

df.to_excel(f"{prefix}_low_freq_actual_results_contextual_BCI_2a_EEG.xlsx")