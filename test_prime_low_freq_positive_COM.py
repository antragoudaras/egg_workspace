from skorch.callbacks import LRScheduler
from torch import nn
from torch.nn import init
import mne
from mne.io import concatenate_raws
from braindecode.util import set_random_seeds
from braindecode.models import ShallowFBCSPNet, Deep4Net
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

from braindecode.preprocessing import (
		exponential_moving_standardize, preprocess, Preprocessor, scale)


def np_to_th(
	X, requires_grad=False, dtype=None, pin_memory=False, **tensor_kwargs
):
	"""
	Convenience function to transform numpy array to `torch.Tensor`.

	Converts `X` to ndarray using asarray if necessary.

	Parameters
	----------
	X: ndarray or list or number
		Input arrays
	requires_grad: bool
		passed on to Variable constructor
	dtype: numpy dtype, optional
	var_kwargs:
		passed on to Variable constructor

	Returns
	-------
	var: `torch.Tensor`
	"""
	if not hasattr(X, "__len__"):
		X = [X]
	X = np.asarray(X)
	if dtype is not None:
		X = X.astype(dtype)
	X_tensor = torch.tensor(X, requires_grad=requires_grad, **tensor_kwargs)
	if pin_memory:
		X_tensor = X_tensor.pin_memory()
	return X_tensor



class Expression(nn.Module):
	"""Compute given expression on forward pass.

	Parameters
	----------
	expression_fn : callable
		Should accept variable number of objects of type
		`torch.autograd.Variable` to compute its output.
	"""

	def __init__(self, expression_fn):
		super(Expression, self).__init__()
		self.expression_fn = expression_fn

	def forward(self, *x):
		return self.expression_fn(*x)

	def __repr__(self):
		if hasattr(self.expression_fn, "func") and hasattr(
			self.expression_fn, "kwargs"
		):
			expression_str = "{:s} {:s}".format(
				self.expression_fn.func.__name__, str(self.expression_fn.kwargs)
			)
		elif hasattr(self.expression_fn, "__name__"):
			expression_str = self.expression_fn.__name__
		else:
			expression_str = repr(self.expression_fn)
		return (
			self.__class__.__name__ +
			"(expression=%s) " % expression_str
		)

class Ensure4d(nn.Module):
	def forward(self, x):
		while(len(x.shape) < 4):
			x = x.unsqueeze(-1)
		return x


def safe_log(x, eps=1e-6):
	""" Prevents :math:`log(0)` by using :math:`log(max(x, eps))`."""
	return torch.log(torch.clamp(x, min=eps))




def square(x):
	return x * x



def transpose_time_to_spat(x):
	"""Swap time and spatial dimensions.

	Returns
	-------
	x: torch.Tensor
		tensor in which last and first dimensions are swapped
	"""
	return x.permute(0, 3, 2, 1)



def squeeze_final_output(x):
	"""Removes empty dimension at end and potentially removes empty time
	 dimension. It does  not just use squeeze as we never want to remove
	 first dimension.

	Returns
	-------
	x: torch.Tensor
		squeezed tensor
	"""

	assert x.size()[3] == 1
	x = x[:, :, :, 0]
	if x.size()[2] == 1:
		x = x[:, :, 0]
	return x



df = pd.read_excel(args.load_dataset, index_col=0)
accuracy = [73.72,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
df['accuracy'] = accuracy



for j in range(25):
	class kostas(nn.Sequential):
		"""Shallow ConvNet model from Schirrmeister et al 2017.
		Model described in [Schirrmeister2017]_.
		Parameters
		----------
		in_chans : int
			XXX
		References
		----------
		.. [Schirrmeister2017] Schirrmeister, R. T., Springenberg, J. T., Fiederer,
		   L. D. J., Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F.
		   & Ball, T. (2017).
		   Deep learning with convolutional neural networks for EEG decoding and
		   visualization.
		   Human Brain Mapping , Aug. 2017.
		   Online: http://dx.doi.org/10.1002/hbm.23730
		"""

		def __init__(
			self,
			in_chans,
			n_classes,
			input_window_samples=None,
			n_filters_time=int(df.iloc[j]['param_1']),
			filter_time_length=int(df.iloc[j]['param_2']),
			n_filters_spat=int(df.iloc[j]['param_3']),
			pool_time_length=int(df.iloc[j]['param_4']),
			pool_time_stride=int(df.iloc[j]['param_5']),
			final_conv_length=30,
			conv_nonlin=square,
			pool_mode="mean",
			pool_nonlin=safe_log,
			split_first_layer=True,
			batch_norm=True,
			batch_norm_alpha=0.1,
			drop_prob=df.iloc[j]['param_6']/10,
		):
			super().__init__()
			if final_conv_length == "auto":
				assert input_window_samples is not None
			self.in_chans = in_chans
			self.n_classes = n_classes
			self.input_window_samples = input_window_samples
			self.n_filters_time = n_filters_time
			self.filter_time_length = filter_time_length
			self.n_filters_spat = n_filters_spat
			self.pool_time_length = pool_time_length
			self.pool_time_stride = pool_time_stride
			self.final_conv_length = final_conv_length
			self.conv_nonlin = conv_nonlin
			self.pool_mode = pool_mode
			self.pool_nonlin = pool_nonlin
			self.split_first_layer = split_first_layer
			self.batch_norm = batch_norm
			self.batch_norm_alpha = batch_norm_alpha
			self.drop_prob = drop_prob

			self.add_module("ensuredims", Ensure4d())
			pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[self.pool_mode]
			if self.split_first_layer:
				self.add_module("dimshuffle", Expression(transpose_time_to_spat))
				self.add_module(
					"conv_time",
					nn.Conv2d(
						1,
						self.n_filters_time,
						(self.filter_time_length, 1),
						stride=1,
					),
				)
				self.add_module(
					"conv_spat",
					nn.Conv2d(
						self.n_filters_time,
						self.n_filters_spat,
						(1, self.in_chans),
						stride=1,
						bias=not self.batch_norm,
					),
				)
				n_filters_conv = self.n_filters_spat
			else:
				self.add_module(
					"conv_time",
					nn.Conv2d(
						self.in_chans,
						self.n_filters_time,
						(self.filter_time_length, 1),
						stride=1,
						bias=not self.batch_norm,
					),
				)
				n_filters_conv = self.n_filters_time
			if self.batch_norm:
				self.add_module(
					"bnorm",
					nn.BatchNorm2d(
						n_filters_conv, momentum=self.batch_norm_alpha, affine=True
					),
				)
			self.add_module("conv_nonlin_exp", Expression(self.conv_nonlin))
			self.add_module(
				"pool",
				pool_class(
					kernel_size=(self.pool_time_length, 1),
					stride=(self.pool_time_stride, 1),
				),
			)
			self.add_module("pool_nonlin_exp", Expression(self.pool_nonlin))
			self.add_module("drop", nn.Dropout(p=self.drop_prob))
			self.eval()
			if self.final_conv_length == "auto":
				out = self(
					np_to_th(
						np.ones(
							(1, self.in_chans, self.input_window_samples, 1),
							dtype=np.float32,
						)
					)
				)
				n_out_time = out.cpu().data.numpy().shape[2]
				self.final_conv_length = n_out_time
			self.add_module(
				"conv_classifier",
				nn.Conv2d(
					n_filters_conv,
					self.n_classes,
					(self.final_conv_length, 1),
					bias=True,
				),
			)
			self.add_module("softmax", nn.LogSoftmax(dim=1))
			self.add_module("squeeze", Expression(squeeze_final_output))

			# Initialization, xavier is same as in paper...
			init.xavier_uniform_(self.conv_time.weight, gain=1)
			# maybe no bias in case of no split layer and batch norm
			if self.split_first_layer or (not self.batch_norm):
				init.constant_(self.conv_time.bias, 0)
			if self.split_first_layer:
				init.xavier_uniform_(self.conv_spat.weight, gain=1)
				if not self.batch_norm:
					init.constant_(self.conv_spat.bias, 0)
			if self.batch_norm:
				init.constant_(self.bnorm.weight, 1)
				init.constant_(self.bnorm.bias, 0)
			init.xavier_uniform_(self.conv_classifier.weight, gain=1)
			init.constant_(self.conv_classifier.bias, 0)


	avgList = []

	for i in range(9):
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

		model = kostas(
			n_chans,
			n_classes,
			input_window_samples=input_window_samples,
			final_conv_length=30,
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
		print("ARCHITECTURE ",j, "SUBJECT ", i+1)
		clf.fit(train_set, y=None, epochs=n_epochs)

		# Extract loss and accuracy values for plotting from history object
		results_columns = ['train_loss', 'valid_loss', 'train_accuracy', 'valid_accuracy']
		dfgtxs = pd.DataFrame(clf.history[:, results_columns], columns=results_columns,
						  index=clf.history[:, 'epoch'])

		avgList.append(100*dfgtxs.valid_accuracy[n_epochs])


	def Average(lst):
		return sum(lst) / len(lst)

	average = Average(avgList)
	print(average)
	df.at[j, 'accuracy'] = average

prefix = None
if "random_dataset_" in args.load_dataset:
	prefix = "random_set"
elif "train_dataset_" in args.load_dataset:
	prefix = "train_set"
elif "val_dataset_" in args.load_dataset:
	prefix = "val_set"

df.to_excel(f"{prefix}_low_freq_actual_results_1000_batch_size_positive_COM.xlsx")