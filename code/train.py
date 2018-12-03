from keras import applications as app
from keras import callbacks as cb
from keras import optimizers as op
from keras import layers
from keras import models

from .unet_build import unet_model_3d
from .loaddata import Load_Data

import os
import numpy as np
import h5py
import argparse
import time
from pprint import pprint

from collections import OrderedDict
import json

class State_Train:
	def __init__(self, json_file):
		# directory
		self.root_dir = os.path.abspath(os.path.join(os.getcwd(),'..'))
		param_dir = os.path.join(self.root_dir + '/training_params/', json_file)
		self.model_dir = os.path.join(self.root_dir, 'model')
		self.log_dir = os.path.join(self.root_dir, 'log')
		# open parameters 
		f = open(param_dir)
		params = json.load(f)
		# parameter
		print('='*100)
		print('Parameters')
		pprint(params)
		print('='*100)
		self.id = len(os.listdir(self.log_dir))
		self.name = params['NAME']
		self.save_name = self.name + '_' + str(self.id)
		self.class_num =  params['CLASS_NUM']
		self.cv_rate = params['CV_RATE']
		self.cv_seed = params['CV_SEED']
		self.epochs = params['EPOCHS']
		self.batch_size = params['BATCH_SIZE']
		self.optimizer = params['OPTIMIZER']

		# model 
		self.model = None
		self.input_shape = (256,256,256,1)

	def run(self):
		# Load image
		LD = Load_Data()
		(ct_images, ct_labels), (mr_images, mr_labels) = LD.train_load()

		# Split
		(ct_train_images, ct_train_labels), (ct_valid_images, ct_valid_labels) = self.data_split(images=ct_images, labels=ct_labels)
		(mr_train_images, mr_train_labels), (mr_valid_images, mr_valid_labels) = self.data_split(images=mr_images, labels=mr_labels)
		del(ct_images, ct_labels, mr_images, mr_labels)

		# Size
		ct_size = (ct_train_images.shape[0], ct_valid_images.shape[0])
		mr_size = (mr_train_images.shape[0], mr_valid_images.shape[0])


		# Data generator
		ct_train = self.data_generator(images=ct_train_images, labels=ct_train_labels, batch_size=self.batch_size)
		ct_val = self.data_generator(images=ct_valid_images, labels=ct_valid_labels, batch_size=self.batch_size)
		mr_train = self.data_generator(images=mr_train_images, labels=mr_train_labels, batch_size=self.batch_size)
		mr_val = self.data_generator(images=mr_valid_images, labels=mr_valid_labels, batch_size=self.batch_size)
		del(ct_train_images, ct_train_labels, ct_valid_images, ct_valid_labels)
		del(mr_train_images, mr_train_labels, mr_valid_images, mr_valid_labels)

		# Check
		print('='*100)
		print('CT')
		print('-'*100)
		print('Train dataset shape: ',ct_train_images.shape)
		print('Validation dataset shape: ',ct_valid_images.shape)
		print('=' * 100)
		print('MR')
		print('-' * 100)
		print('Train dataset shape: ', mr_train_images.shape)
		print('Validation dataset shape: ', mr_valid_images.shape)

		# model
		self.model_(input_shape=self.input_shape)

		# train
		history, train_time = self.training(train=mr_train, val=mr_val, size=mr_size)

		# report
		self.report_json(history=history, time=train_time)


	def data_split(self, images, labels):
		'''
		:param images: Images
		:param labels: Labels
		:return: (train_images, train_labels), (valid_images, valid_labels)
		'''
		img_cnt = images.shape[0]
		train_cnt = np.ceil(img_cnt*self.cv_rate)

		idx = np.arange(img_cnt)

		np.random.seed(self.cv_seed)
		idx = np.random.shuffle(idx)

		train_idx = idx[:train_cnt]
		valid_idx = idx[train_cnt:]

		train_images = images[train_idx]
		train_labels = labels[train_idx]
		valid_images = images[valid_idx]
		valid_labels = labels[valid_idx]

		return (train_images, train_labels), (valid_images, valid_labels)


	def data_generator(self, images, labels, batch_size):
		'''
		:param images: [x1,x2,x3,...]
		:param labels: [y1,y2,y3,...]
		:param batch_size: Number of batch size
		:return: if batch_size is 3, then yield [x1,x2,x3], [y1,y2,y3]
		'''

		img_cnt = images.shape[0]
		steps_size = img_cnt // batch_size

		for i in steps_size:
			batch_x, batch_y = images[batch_size*i:batch_size*(i+1)], labels[batch_size*i:batch_size*(i+1)]
			yield batch_x, batch_y



	def model_(self, input_shape):
		print('='*100)
		print('Load Model')
		print('Model name: ', self.save_name)
		print('-'*100)

		self.model = unet_model_3d(input_shape,
									n_labels=self.class_num,
									initial_learning_rate=0.00001,
									n_base_filters=32,
									include_label_wise_dice_coefficients=True,
									batch_normalization=False,
									activation_name="relu")

		print('Complete')
		print('='*100)
			

	def training(self, train, val, size, weights=None):
		print('='*100)
		print('Start training')
		print('-'*100)
		if self.self_training:
			if 'selftraining' not in self.save_name:
				self.save_name += '_selftraining'
				
		ckp = cb.ModelCheckpoint(self.model_dir + '/' + self.save_name + '.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto', period=1)
		es = cb.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
		# rlp = cb.ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=5, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.000001)
		start = time.time()
		history = self.model.fit_generator(generator=train,
											steps_per_epoch=size[0]//self.batch_size,
											epochs=self.epochs,
											validation_data=val,
											validation_steps=size[1]//self.batch_size,
											class_weight=weights,
											callbacks=[es,ckp]
						   )
		e = int(time.time() - start)
		print('-'*100)
		print('Complete')
		print('-'*100)
		train_time = '{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60)
		print('Train time: ',train_time)
		print('='*100)
		return history, train_time


	def report_json(self, history, time):
		log = OrderedDict()
		m_compile = OrderedDict()
		log['ID'] = self.id
		log['NAME'] = self.name
		log['TIME'] = time
		log['INPUT_SHAPE'] = self.model.input_shape
		m_compile['OPTIMIZER'] = str(type(self.model.optimizer))
		m_compile['LOSS'] = self.model.loss
		m_compile['METRICS'] = self.model.metrics_names
		log['COMPILE'] = m_compile
		h = history.params
		h['LOSS'] = history.history['loss']
		h['ACCURACY'] = history.history['acc']
		h['VAL_LOSS'] = history.history['val_loss']
		h['VAL_ACCURACY'] = history.history['val_acc']
		log['HISTORY'] = h

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

	ST = State_Train(json_file=args.params)
	ST.run()