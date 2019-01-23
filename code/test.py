from loaddata import Load_Data
from metrics import *

import os
import numpy as np
import nibabel as nib
import argparse
import time
import h5py
from pprint import pprint
from tqdm import tqdm
from functools import partial

from keras import models

from collections import OrderedDict
import json

class MMWHS_Test:
	def __init__(self, args):
		self.model_name = args.model_name
		self.dataset = args.dataset
		self.input_channel = args.input_channel
		self.image_size = args.image_size

		self.root_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
		self.model_dir = os.path.join(self.root_dir, 'model')
		self.test_dir = os.path.join(self.root_dir, 'dataset/h5py/train_hf' + str(self.image_size) + '_7')
		self.save_dir = os.path.join(self.root_dir, 'predict_image')

		self.model = None
		self.test_image_num = 20

		self.test_hf = h5py.File(self.test_dir, 'r')
		print('='*100)
		print('test_hf: ',list(self.test_hf.keys()))
		print('=' * 100)

	def run(self):
		# Load model
		self.model_()
		print(self.model.summary())
		# predict
		print('=' * 100)
		print('Start to predict label')
		for i in range(self.test_image_num):
			# Create generator
			test_image = self.load_data(i)
			prob = self.model.predict(test_image, verbose=1)
			prob = np.moveaxis(prob, 1, -1)[0]
			pred = np.round(prob).astype(int)
			print('Predict image shape: ',pred.shape)
			print(np.sum(pred))
			print('Complete')
			# Save images
			self.save_predict_value(pred, i)
		self.test_hf.close()


	def load_data(self, idx):
		"""
        Generator to yield inputs in batches.
        """
		img = np.array(self.test_hf['{}_image_{}'.format(self.dataset.lower(), idx)]).reshape((-1,) + (self.image_size,) * 3 + (1,))
		img = np.moveaxis(img, -1, 1)
		return img


	def model_(self):
		print('='*100)
		print('Load model')
		custom_objects = {'weighted_dice_coefficient_loss': weighted_dice_coefficient_loss,'dice_coefficient': dice_coefficient}

		if self.input_channel > 1:
			for label_index in range(self.input_channel):
				custom_objects['DSC_{0}'.format(label_index)] = get_label_dice_coefficient_function(label_index)
		print(custom_objects.keys())

		self.model = models.load_model(self.model_dir + '/' + self.model_name + '.h5', custom_objects=custom_objects)

		print('Complete')


	def save_predict_value(self, pred, idx):
		print('='*100)
		print('Save predict images')
		pred_image = nib.Nifti1Image(pred, affine=np.eye(4))
		nib.save(pred_image, self.save_dir + '/{}_train_pred_{}.nii'.format(self.dataset, idx))
		print('Complete')


if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str, help='CT or MR')
	parser.add_argument('--model_name', type=str, help='Model name')
	parser.add_argument('--input_channel', type=int, default=8, help='Number of class')
	parser.add_argument('--image_size', type=int, default=128, help='Input size')
	args = parser.parse_args()

	t = MMWHS_Test(args)
	t.run()


