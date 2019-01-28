from keras import callbacks as cb

from loaddata import *
from models import unet_3d, unet_2d

import os
import numpy as np
import argparse
import time
from pprint import pprint

from collections import OrderedDict
import json

class MMWHS_Train:
	def __init__(self, json_file):
		# directory
		self.root_dir = os.path.abspath(os.path.join(os.getcwd(),'..'))
		param_dir = os.path.join(self.root_dir + '/training_params/', json_file)
		self.model_dir = os.path.join(self.root_dir, 'model')
		self.log_dir = os.path.join(self.root_dir, 'log')
		self.train_dir = os.path.abspath(os.path.join(self.root_dir, 'dataset/ct_train_test/ct_train'))
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
		self.patch_dim = params['PATCH_DIM']
		self.resize_r = params['RESIZE_R']
		self.epochs = params['EPOCHS']
		self.output_channel = params['OUTPUT_CHANNEL']
		self.batch_size = params['BATCH_SIZE']
		self.optimizer = params['OPTIMIZER']
		self.learning_rate = params['LEARNINGRATE']

		self.id = len([name for name in os.listdir(self.log_dir) if self.name in name])
		self.save_name = self.name + '_' + str(self.id)
		# model 
		self.model = None



	def run(self):
		# Create generator

		# split train and validation
		valid_subjects = list()
		train_subjects = list()
		for i in range(0,21):
			if i % 4 == 0 and i != 0 :
				valid_subjects.append(i)
			else:
				train_subjects.append(i)

		size = [len(train_subjects), len(valid_subjects)]
		print('Number of train subjects: ',size[0])
		print('Number of valid subjects: ',size[1])
		train_gen = data_gen(dir=self.train_dir, batch_size=self.batch_size, patch_dim=self.patch_dim, resize_r=self.resize_r, subjects=train_subjects)
		valid_gen = data_gen(dir=self.train_dir, batch_size=self.batch_size, patch_dim=self.patch_dim, resize_r=self.resize_r, subjects=valid_subjects)

		# model
		self.model_()
		print(self.model.summary())

		# train
		history, train_time = self.training(train=train_gen, valid=valid_gen, size=size)

		# report
		self.report_json(history=history, time=train_time)



	def model_(self):
		print('='*100)
		print('Load Model')
		print('Model name: ', self.save_name)
		print('-'*100)

		model = unet_3d.Unet3d(input_size=self.patch_dim,
									lrate=self.learning_rate,
									n_labels=self.output_channel,
									n_base_filters=32,
									batch_normalization=False,
									deconvolution=True)
		self.model = model.build()

		print('Complete')
		print('='*100)
			

	def training(self, train, valid, size, weights=None):
		print('='*100)
		print('Start training')
		print('-'*100)

		ckp = cb.ModelCheckpoint(self.model_dir + '/' + self.save_name + '.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto', period=1)
		es = cb.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
		rlp = cb.ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=5, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.000001)
		tb = cb.TensorBoard(log_dir='../tensor_board', histogram_freq=0, batch_size=1, write_graph=True, write_grads=True, write_images=True,  update_freq='epoch')
		start = time.time()

		history = self.model.fit_generator(generator=train,
											steps_per_epoch=size[0]//self.batch_size,
											epochs=self.epochs,
											validation_data=valid,
											validation_steps=size[1]//self.batch_size,
											class_weight=weights,
										    verbose=1,
											callbacks=[ckp,tb])
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
		log['DATA'] = self.data
		log['PATCH_DIM'] = self.patch_dim
		log['RESIZE_R'] = self.resize_r
		log['EPOCHS'] = self.epochs
		log['BATCH_SIZE'] =self.batch_size
		log['LEARNINGRATE'] = self.learning_rate

		m_compile['OPTIMIZER'] = str(type(self.model.optimizer))
		m_compile['LOSS'] = str(self.model.loss)
		m_compile['METRICS'] = str(self.model.metrics_names)
		log['COMPILE'] = m_compile

		h = history.params
		h['LOSS'] = history.history['loss']
		h['DSC_TOTAL'] = history.history['dice_coefficient']
		h['VAL_LOSS'] = history.history['val_loss']
		h['VAL_DSC_TOTAL'] = history.history['val_dice_coefficient']
		h['DSC0'] = history.history['DSC_0']
		h['DSC1'] = history.history['DSC_1']
		h['DSC2'] = history.history['DSC_2']
		h['DSC3'] = history.history['DSC_3']
		h['DSC4'] = history.history['DSC_4']
		h['DSC5'] = history.history['DSC_5']
		h['DSC6'] = history.history['DSC_6']
		h['DSC7'] = history.history['DSC_7']
		h['VAL_DSC0'] = history.history['val_DSC_0']
		h['VAL_DSC1'] = history.history['val_DSC_1']
		h['VAL_DSC2'] = history.history['val_DSC_2']
		h['VAL_DSC3'] = history.history['val_DSC_3']
		h['VAL_DSC4'] = history.history['val_DSC_4']
		h['VAL_DSC5'] = history.history['val_DSC_5']
		h['VAL_DSC6'] = history.history['val_DSC_6']
		h['VAL_DSC7'] = history.history['val_DSC_7']
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

	ST = MMWHS_Train(json_file=args.params)
	ST.run()