from keras import callbacks as cb

from loaddata import Load_Data
from models import unet_3d, unet_2d
from keras.preprocessing.image import ImageDataGenerator

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
		self.class_num =  params['CLASS_NUM']
		self.image_size = params['IMAGE_SIZE']
		self.cv_rate = params['CV_RATE']
		self.cv_seed = params['CV_SEED']
		self.epochs = params['EPOCHS']
		self.batch_size = params['BATCH_SIZE']
		self.optimizer = params['OPTIMIZER']
		self.learning_rate = params['LEARNINGRATE']
		self.dimension = params['DIMENSION']

		self.id = len([name for name in os.listdir(self.log_dir) if self.name in name])
		self.save_name = self.name + '_' + str(self.id) + '_' + str(self.image_size) + '_' + str(self.class_num)

		# model 
		self.model = None
		if self.dimension == '2D':
			self.input_shape = ((1,) + (self.image_size ,)*2)
		elif self.dimension == '3D':
			self.input_shape = ((1,) + (self.image_size ,)*3)

	def run(self):
		# Create generator
		LD = Load_Data(dataset=self.data, channel=self.class_num, dimension=self.dimension, size=self.image_size)

		# split train and validation
		subjects = list(range(20)) # 20 is Number of subjects
		choice = np.random.choice(subjects, size=int(20*self.cv_rate), replace=False)
		train_subjects = list(map(lambda x: subjects.pop(x), sorted(choice, key=lambda x: -x)))
		valid_subjects = subjects

		size = [len(train_subjects), len(valid_subjects)]
		print('Number of train subjects: ',size[0])
		print('Number of valid subjects: ',size[1])
		train_gen = LD.data_gen(batch_size=self.batch_size, subjects=train_subjects)
		valid_gen = LD.data_gen(batch_size=self.batch_size, subjects=valid_subjects)

		# model
		self.model_()
		print(self.model.summary())

		# train
		history, train_time = self.training(train=train_gen, valid=valid_gen, size=size)



		# report
		self.report_json(history=history, time=train_time)

	def data_split(self, images, labels):
		'''
		:param images: Images
		:param labels: Labels
		:return: (train_images, train_labels), (valid_images, valid_labels)
		'''
		subjects = 20
		train_cnt = int(subjects*self.cv_rate)
		print('-'*100)
		print('Number of train subjects: ',train_cnt)
		print('Number of validation subjects: ',subjects - train_cnt)

		if self.dimension=='2D':
			subjects *= self.image_size
			train_cnt *= self.image_size

		idx = np.arange(subjects)

		train_idx = idx[:train_cnt]
		valid_idx = idx[train_cnt:]

		train_images = images[train_idx]
		train_labels = labels[train_idx]
		valid_images = images[valid_idx]
		valid_labels = labels[valid_idx]

		return (train_images, train_labels), (valid_images, valid_labels)


	def model_(self):
		print('='*100)
		print('Load Model')
		print('Model name: ', self.save_name)
		print('-'*100)
		if self.name == 'UNET_3D':
			model = unet_3d.Unet3d(input_shape=self.input_shape,
							n_labels=self.class_num,
							initial_learning_rate=self.learning_rate,
							n_base_filters=64,
							batch_normalization=False,
							deconvolution=True)
		elif self.name == 'UNET_2D':
			model = unet_2d.Unet2d(input_shape=self.input_shape,
								   n_labels=self.class_num,
								   initial_learning_rate=self.learning_rate)
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
		# history = self.model.fit_generator(x=X, y=Y,
		# 						batch_size=self.batch_size,
		# 						epochs=self.epochs,
		# 						verbose=1,
		# 						callbacks=[ckp,tb],
		# 						validation_data=(ValX, ValY),
		# 						shuffle=True,
		# 						class_weight=weights)
		if self.dimension=='2D':
			size[0] = size[0]*self.image_size
			size[1] = size[1]*self.image_size

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
		log['INPUT_SHAPE'] = self.model.input_shape
		log['DATA'] = self.data
		log['CLASS_NUM'] = self.class_num
		log['IMAGE_SIZE'] = self.image_size
		log['CV_RATE'] = self.cv_rate
		log['CV_SEED'] = self.cv_seed
		log['EPOCHS'] = self.epochs
		log['BATCH_SIZE'] =self.batch_size
		log['LEARNINGRATE'] = self.learning_rate
		log['DIMENSION'] = self.dimension

		m_compile['OPTIMIZER'] = str(type(self.model.optimizer))
		m_compile['LOSS'] = str(self.model.loss)
		m_compile['METRICS'] = str(self.model.metrics_names)
		log['COMPILE'] = m_compile

		h = history.params
		h['LOSS'] = history.history['loss']
		h['DSC_TOTAL'] = history.history['dice_coefficient']
		h['VAL_LOSS'] = history.history['val_loss']
		h['VAL_DSC_TOTAL'] = history.history['val_dice_coefficient']
		if self.class_num > 1:
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