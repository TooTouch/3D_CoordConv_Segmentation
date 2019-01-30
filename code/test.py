import os
import numpy as np
import nibabel as nib
import argparse
import time
import h5py
from glob import glob
from pprint import pprint
from tqdm import tqdm
from keras import models

from collections import OrderedDict
import json

from utils import *
from metrics import *


class MMWHS_Test:
	def __init__(self, model_id):
		self.model_id = model_id

		self.root_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
		self.test_dir = os.path.join(self.root_dir, 'dataset/ct_train_test/ct_train')
		self.model_dir = os.path.join(self.root_dir, 'model')
		self.save_dir = os.path.join(self.root_dir, 'predict_image/{}'.format(self.model_id))
		self.log_dir = os.path.join(self.root_dir, 'log')

		f = open('{}/{}_info.json'.format(self.log_dir, self.model_id))
		j = json.load(f)

		# self.patch_dim = j['PATCH_DIM']
		# self.resize_r = j['RESIZE_R']
		# self.loss = j['COMPILE']['LOSS'].split()[1]
		# self.output_channel = j['OUTPUT_CHANNEL']
		self.rename_map = [0, 205, 420, 500, 550, 600, 820, 850]

		self.patch_dim = 96
		self.resize_r = 0.6
		self.loss = softmax_weighted_loss
		self.output_channel = 8
		self.ovlp_ita = 4



		self.model = None
		self.test_image_num = 40



	def run(self):
		# Load model
		self.model_()
		print(self.model.summary())
		# predict
		print('=' * 100)
		print('Start to predict label')
		test_list = glob('{}/*.nii/*.nii'.format(self.test_dir))

		for i in tqdm(range(0,self.test_image_num,2)):

			test_i = nib.load(test_list[i])
			test_i_affine = test_i.affine
			test_image = test_i.get_data().copy()

			resize_dim = (np.array(test_image.shape) * self.resize_r).astype('int')
			test_image_resize = resize(test_image, resize_dim, order=1, preserve_range=True, mode='constant')

			test_image_resize = test_image_resize.astype('float32')
			test_image_resize /= 255.0

			cube_list = decompose_vol2cube(test_image_resize, 1, self.patch_dim, self.ovlp_ita)
			cube_label_list = []
			for c in tqdm(range(len(cube_list))):
				cube2test = cube_list[c]
				mean_temp = np.mean(cube2test)
				std_temp = np.std(cube2test)
				cube2test_norm = (cube2test - mean_temp) / std_temp

				cube_label = self.model.predict(cube2test_norm, verbose=0)
				cube_label = np.argmax(cube_label, axis=-1)
				cube_label_list.append(cube_label)


			composed_orig = compose_label_cube2vol(cube_label_list, resize_dim, self.patch_dim, self.ovlp_ita, self.output_channel)
			composed_label = np.zeros(composed_orig.shape, dtype='int16')

			for j in range(len(self.rename_map)):
				composed_label[composed_orig==j] = self.rename_map[j]
			composed_label = composed_label.astype('int16')

			print('Labels check: ',np.unique(composed_label))
			print('Labels shape: ',composed_label.shape)

			composed_label_resize = resize(composed_label, test_image.shape, order=0, preserve_range=True, mode='constant')
			composed_label_resize = composed_label_resize.astype('int16')

			print('Resize labels check: ',np.unique(composed_label_resize))
			print('Resize labels shape: ',composed_label_resize.shape)

			print('=' * 100)
			print('Save predict images')
			try:
				if not(os.path.isdir(self.save_dir)):
					os.mkdir(self.save_dir)
			except:
				print('Fail to create directory.')
			labeling_image = nib.Nifti1Image(composed_label_resize, affine=test_i_affine)
			nib.save(labeling_image, '{}/{}_{}.nii'.format(self.save_dir, self.model_id, i))
			print('Complete')


	def model_(self):
		print('='*100)
		print('Load model')
		custom_objects = {'%s'%str(self.loss).split()[1] : self.loss, 'dice_coefficient' : dice_coefficient}

		if self.output_channel > 1:
			for label_index in range(self.output_channel):
				custom_objects['DSC_{0}'.format(label_index)] = get_label_dice_coefficient_function(label_index)
		print(custom_objects.keys())
		self.model = models.load_model('{}/{}.h5'.format(self.model_dir, self.model_id), custom_objects=custom_objects)
		print('Complete')



if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_id', type=str, help='model name')
	args = parser.parse_args()

	t = MMWHS_Test(args.model_id)
	t.run()


