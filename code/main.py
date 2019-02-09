from keras import callbacks as cb
from keras import models

from loaddata import *
from models import unet_3d

import os
import numpy as np
import argparse
import time
import nibabel as nib
from tqdm import tqdm
from skimage.transform import resize
from glob import glob
from pprint import pprint

from collections import OrderedDict
import json

from metrics import *
from utils import *

class MMWHS_Train:
	def __init__(self, json_file):
		# directory
		self.root_dir = os.path.abspath(os.path.join(os.getcwd(),'..'))
		param_dir = os.path.join(self.root_dir + '/training_params/', json_file)
		self.model_dir = os.path.join(self.root_dir, 'model')
		self.log_dir = os.path.join(self.root_dir, 'log')
		self.train_dir = os.path.abspath(os.path.join(self.root_dir, 'dataset/ct_train_test/ct_train'))
		self.test_dir = os.path.abspath(os.path.join(self.root_dir, 'dataset/ct_train_test/ct_test'))

		# open parameters 
		f = open(param_dir)
		params = json.load(f)
		# parameter
		print('='*100)
		print('Parameters')
		pprint(params)
		print('='*100)

		self.name = params['NAME']
		self.data = params['DATA']
		self.image_num = params['IMAGE_NUM']
		self.patch_dim = params['PATCH_DIM']
		self.train_patch_num = params['TRAIN_PATCH_NUM']
		self.valid_patch_num = params['VALID_PATCH_NUM']
		self.resize_r = params['RESIZE_R']
		self.epochs = params['EPOCHS']
		self.input_channel = params['INPUT_CHANNEL']
		self.output_channel = params['OUTPUT_CHANNEL']
		self.batch_size = params['BATCH_SIZE']
		self.optimizer = params['OPTIMIZER']
		self.learning_rate = params['LEARNINGRATE']
		self.normalization = None if params['NORMALIZATION']==0 else params['NORMALIZATION']
		self.coordnet = bool(params['COORD_NET'])
		self.image_coord = bool(params['IMAGE_COORD'])
		self.loss = softmax_weighted_loss
		self.ovlp_ita = params['OVERLAP_FACTOR']
		self.multi_gpu = params['MULTI_GPU']
		self.gpus = ','.join(self.multi_gpu)
		self.rename_map = [0, 205, 420, 500, 550, 600, 820, 850]

		# create directory
		if not (os.path.isdir(self.model_dir)):
			os.makedirs(self.model_dir)
		if not (os.path.isdir(self.log_dir)):
			os.makedirs(self.log_dir)
		if not (os.path.isdir('../history')):
			os.makedirs('../history')

		# input channel
		if self.image_coord:
			self.input_channel += 3

		# save
		self.id = len([name for name in os.listdir(self.log_dir) if self.name in name])
		self.save_name = self.name + '_' + str(self.id)
		self.save_dir = os.path.join(self.root_dir, 'predict_image/{}'.format(self.save_name))
		if not (os.path.isdir(self.save_dir)):
			os.makedirs(self.save_dir)

		# model
		self.model = None

		# allow_gpu
		os.environ["CUDA_VISIBLE_DEVICES"]=self.gpus

	def run(self):
		self.train()
		self.test()

	def train(self):
		# split train and validation
		patients = np.arange(20)[:2]
		train_patients = patients[:1]
		valid_patients = patients[-1:]

		size = [len(train_patients), len(valid_patients)]
		print('Number of train patients: ',size[0])
		print('train patients: ',train_patients)
		print('Number of valid patients: ',size[1])
		print('validatiion patients: ',valid_patients)

		pair_list = glob('{}/*.nii/*.nii'.format(self.train_dir))

		# output : resized images, resized labels
		imgs, labels = load_data_pairs(pair_list[:4], self.resize_r, self.output_channel, self.rename_map)

		# Create generator
		train_gen = trainGenerator(imgs=imgs,
									 labels=labels,
									 patients=train_patients,
									 batch_size=self.batch_size,
									 patch_dim=self.patch_dim,
									 patch_size=self.train_patch_num,
									 output_chn=self.output_channel,
								     input_chn=self.input_channel,
								     coordnet=self.image_coord)

		valid_gen = validGenerator(imgs=imgs,
									 labels=labels,
									 patients=valid_patients,
									 batch_size=self.batch_size,
									 patch_dim=self.patch_dim,
									 patch_size=self.valid_patch_num,
									 output_chn=self.output_channel)

		# model
		self.model_()
		print(self.model.summary())

		# train
		history, train_time = self.fit(train=train_gen, valid=valid_gen, size=size)

		# report
		self.report_json(time=train_time)

		# del
		del train_gen, valid_gen


	def test(self):
		# Load model
		self.saved_model_()
		print(self.model.summary())

		# predict
		print('=' * 100)
		print('Start to predict label')
		test_list = glob('{}/*.nii/*.nii'.format(self.test_dir))

		for i in tqdm(range(len(test_list))):

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
				# mean_temp = np.mean(cube2test)
				# std_temp = np.std(cube2test)
				# cube2test_norm = (cube2test - mean_temp) / std_temp

				cube_label = self.model.predict(cube2test, verbose=0)
				cube_label = np.argmax(cube_label, axis=-1)
				cube_label_list.append(cube_label)


			composed_orig = compose_label_cube2vol(cube_label_list, resize_dim, self.patch_dim, self.ovlp_ita, self.output_channel)
			composed_label = np.zeros(composed_orig.shape, dtype='int16')

			for j in range(len(self.rename_map)):
				composed_label[composed_orig==j] = self.rename_map[j]
			composed_label = composed_label.astype('int16')

			composed_label_resize = resize(composed_label, test_image.shape, order=0, preserve_range=True, mode='constant')
			composed_label_resize = composed_label_resize.astype('int16')

			print('=' * 100)
			print('Save predict images')
			try:
				if not(os.path.isdir(self.save_dir)):
					os.mkdir(self.save_dir)
			except:
				print('Fail to create directory.')
			labeling_image = nib.Nifti1Image(composed_label_resize, affine=test_i_affine)
			nib.save(labeling_image, '{}/{}_{}.nii'.format(self.save_dir, self.save_name, i))
			print('Complete')



	def model_(self):
		print('='*100)
		print('Load Model')
		print('Model name: ', self.save_name)
		print('-'*100)

		model = unet_3d.Unet3d(input_size=self.patch_dim,
								lrate=self.learning_rate,
								n_labels=self.output_channel,
								n_base_filters=32,
								normalization=self.normalization,
								deconvolution=True,
								coordnet=self.coordnet)

		self.model = model.build(input_chn=self.input_channel, multi_gpu=self.multi_gpu)

		print('Complete')
		print('='*100)


	def saved_model_(self):
		print('='*100)
		print('Load model')
		custom_objects = {'%s'%str(self.loss).split()[1] : self.loss, 'dice_coefficient' : dice_coefficient}

		if self.output_channel > 1:
			for label_index in range(self.output_channel):
				custom_objects['DSC_{0}'.format(label_index)] = get_label_dice_coefficient_function(label_index)
		print(custom_objects.keys())
		self.model = models.load_model('{}/{}.h5'.format(self.model_dir, self.save_name), custom_objects=custom_objects)
		print('Complete')


	def fit(self, train, valid, size, weights=None):
		print('='*100)
		print('Start training')
		print('-'*100)

		ckp = cb.ModelCheckpoint(self.model_dir + '/' + self.save_name + '.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto', period=1)
		# rlp = cb.ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=5, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.000001)
		csv_logger = cb.CSVLogger('../history/{}.csv'.format(self.save_name))
		# tb = TensorBoardWrapper(valid, nb_steps=size[1]*self.valid_patch_num, log_dir='./tensor_board/{}'.format(self.save_name), histogram_freq=1, batch_size=1, write_graph=True, write_grads=True, write_images=False)
		callbacks = [ckp,csv_logger]

		start = time.time()
		history = self.model.fit_generator(generator=train,
											steps_per_epoch=size[0]*self.train_patch_num,
											epochs=self.epochs,
											validation_data=valid,
											validation_steps=size[1]*self.valid_patch_num,
											class_weight=weights,
											verbose=1,
											callbacks=callbacks)
		e = int(time.time() - start)
		print('-'*100)
		print('Complete')
		print('-'*100)
		train_time = '{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60)
		print('Train time: ',train_time)
		print('='*100)
		return history, train_time



	def report_json(self, time):
		log = OrderedDict()
		m_compile = OrderedDict()
		log['ID'] = self.id
		log['NAME'] = self.name
		log['TIME'] = time
		log['DATA'] = self.data
		log['PATCH_DIM'] = self.patch_dim
		log['RESIZE_R'] = self.resize_r
		log['EPOCHS'] = self.epochs
		log['BATCH_SIZE'] =self.batch_size
		log['LEARNINGRATE'] = self.learning_rate
		log['OUTPUT_CHANNEL'] = self.output_channel

		m_compile['OPTIMIZER'] = str(type(self.model.optimizer))
		m_compile['LOSS'] = str(self.model.loss)
		m_compile['METRICS'] = str(self.model.metrics_names)
		log['COMPILE'] = m_compile

		print('='*100)
		print('Save log to json file')
		with open(self.log_dir + '/' + self.save_name + '_info.json','w',encoding='utf-8') as make_file:
			json.dump(log, make_file, ensure_ascii=False, indent='\t')
		print('Complete')
		print('='*100)


if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--params', type=str, help='Json file name containig paramters')
	args = parser.parse_args()

	ST = MMWHS_Train(json_file=args.params)
	ST.run()