# from keras import callbacks as cb
#
# from loaddata import Load_Data
# #from models import unet_3d, unet_2d

import os
import numpy as np
import argparse
import time
from pprint import pprint

from collections import OrderedDict
import json

# Pytorch_loader

from pytorch_loader import CardiacDataset
from torch.utils.data import Dataset ,DataLoader

from torchsummary import summary
from torch.autograd import Variable
import torch

from utils_pytorch import compute_per_channel_dice


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
		self.class_num = params['CLASS_NUM']
		self.input_channel = params['INPUT_CHANNEL']
		self.image_size = params['IMAGE_SIZE']
		self.dimension = params['DIMENSION']
		self.cv_rate = params['CV_RATE']
		self.cv_seed = params['CV_SEED']
		self.batch_size = params['BATCH_SIZE']
		self.epochs = params['EPOCHS']
		self.optimizer = params['OPTIMIZER']
		self.learning_rate = params['LEARNINGRATE']



		## Pytorch
		self.loss = params['LOSS']
		self.save_model_dr = os.path.join(self.root_dir + '/saved_models/', params['NAME'])



		self.id = len([name for name in os.listdir(self.log_dir) if self.name in name])
		self.save_name = self.name + '_' + str(self.id) + '_' + str(self.image_size) + '_' + str(self.class_num)

		# model 
		self.model = None

		self.input_shape = ((1,) + (self.image_size ,)*3)


	def run(self):

		# Create generator
		# 1 ~ 18 Train
		# 19 ~ 20 Validation

		dataset = CardiacDataset(dataset='CT', class_num=self.class_num, channel=self.input_channel, dimension=self.dimension,
								 size=self.image_size, subjects=[i for i in range(18)])

		valid_dataset = CardiacDataset(dataset='CT', class_num=self.class_num, channel=self.input_channel, dimension=self.dimension,
									   size=self.image_size, subjects=[i for i in range(18, 20)])

		train_loader = DataLoader(dataset=dataset,
								  batch_size=1,
								  shuffle=True,
								  num_workers=0)

		valid_loader = DataLoader(dataset=valid_dataset,
								  batch_size=1,
								  shuffle=True,
								  num_workers=0)
		# model
		self.model_()
		summary(self.model, self.input_shape)
		# train
		history, train_time = self.training(train_loader=train_loader, valid_loader=valid_loader)
		# report
		self.report_json(history=history, time=train_time)


	# def data_split(self, images, labels):
	# 	'''
	# 	:param images: Images
	# 	:param labels: Labels
	# 	:return: (train_images, train_labels), (valid_images, valid_labels)
	# 	'''
	# 	subjects = 20
	# 	train_cnt = int(subjects*self.cv_rate)
	# 	print('-'*100)
	# 	print('Number of train subjects: ',train_cnt)
	# 	print('Number of validation subjects: ',subjects - train_cnt)
	#
	# 	idx = np.arange(subjects)
	#
	# 	train_idx = idx[:train_cnt]
	# 	valid_idx = idx[train_cnt:]
	#
	# 	train_images = images[train_idx]
	# 	train_labels = labels[train_idx]
	# 	valid_images = images[valid_idx]
	# 	valid_labels = labels[valid_idx]
	#
	# 	return (train_images, train_labels), (valid_images, valid_labels)


	def model_(self):
		print('='*100)
		print('Load Model')
		print('Model name: ', self.save_name)
		print('-'*100)

		if self.name == "UNET_3D_PT":
			from models.pytorch_unet3d import unet3d
			model = unet3d(n_labels=self.input_channel).cuda()

		elif self.name == "UNET_3D_SE_PT":
			from models.pytorch_unet3d_se import unet3d_se
			model = unet3d_se(n_labels=self.input_channel).cuda()

		elif self.name == "UNET_3D_SE_ALL_PT":
			from models.pytorch_unet3d_se_all import unet3d_se_all
			model = unet3d_se_all(n_labels=self.input_channel).cuda()


		device = torch.device('cuda:0')
		model.to(device)

		# model = unet_3d.Unet3d(input_shape=self.input_shape,
		# 				n_labels=self.input_channel,
		# 				lrate=self.learning_rate,
		# 				n_base_filters=32,
		# 				batch_normalization=False,
		# 				deconvolution=True)
		#

		if 'finetune' in self.name:
			self.model = model.finetune_model()
		else:
			self.model = model

		print('Complete')
		print('='*100)
			

	def training(self, train_loader, valid_loader, weights=None):
		print('='*100)
		print('Start training')
		print('-'*100)

		history = {}
		history['loss'] = []
		history['dice_coefficient'] = []
		history['val_loss'] = []
		history['val_dice_coefficient'] = []

		# Making Diretorcy
		model_path = self.save_model_dr

		try:
			os.mkdir(model_path)
		except:
			pass



		start = time.time()

		if self.loss == 'CrossEntropy':
			loss_fn = torch.nn.CrossEntropyLoss(reduction='elementwise_mean')

		else:
			loss_fn = torch.nn.CrossEntropyLoss(reduction='elementwise_mean')

		## model load
		model = self.model

		learning_rate = self.learning_rate

		if self.optimizer == 'adam':
			optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
		else :
			optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


		print("Total Epochs:", self.epochs)
		for t in range(self.epochs):

			print("=" * 50)
			print("Starting Epoch", t)

			train_loss = 0
			train_dsc = []

			## Training


			for idx, data in enumerate(train_loader):
				img, lab = data
				lab = np.argmax(lab, axis=-1)
				lab = lab.reshape(64, 64, 64, 1)
				img = img.transpose(-1, 1)
				lab = lab.transpose(-1, 0)

				x = Variable(img).float().cuda()
				y = Variable(lab).long().cuda()

				y_pred = model(x)
				loss = loss_fn(y_pred, y)
				dsc = compute_per_channel_dice(y_pred, y)
				dsc_list = dsc.tolist()

				train_loss += loss.data.tolist()
				train_dsc.append(dsc_list)

				#         history['loss'].append(loss.data.item())
				#         history['dsc'].append(dsc_list)

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

			if t % 5 == 0:
				print(t, loss.data.item())
				print(
					"DSC 0 : {}, DSC 1 : {}, DSC 2 : {}, DSC 3 : {}, DSC 4 : {}, DSC 5 : {}, DSC 6 : {}, DSC 7 : {}".format(
						dsc_list[0], dsc_list[1], dsc_list[2], dsc_list[3], dsc_list[4], dsc_list[5], dsc_list[6],
						dsc_list[7]))

			if t % 50 == 0:
				model_fname = 'baseline_64_{}.pt'.format(t)
				print('=' * 50)
				print("Saving Model in epcoh", t)
				torch.save(model.state_dict(), os.path.join(model_path, model_fname))

			history['loss'].append(train_loss / (idx + 1))
			history['dice_coefficient'].append(np.mean(train_dsc, axis=0).tolist())

			## Validation

			val_loss = 0
			val_dsc = []

			for val_idx, val_data in enumerate(valid_loader):
				val_img, val_lab = val_data
				val_lab = np.argmax(val_lab, axis=-1)
				val_lab = val_lab.reshape(64, 64, 64, 1)
				val_img = val_img.transpose(-1, 1)
				val_lab = val_lab.transpose(-1, 0)

				x = Variable(val_img).float().cuda()
				y = Variable(val_lab).long().cuda()

				y_pred = model(x)
				loss = loss_fn(y_pred, y)
				val_loss += loss.data.tolist()

				dsc = compute_per_channel_dice(y_pred, y)
				dsc_list = dsc.tolist()
				val_dsc.append(dsc_list)

				del x, y

			history['val_loss'].append(val_loss / (val_idx + 1))
			history['val_dice_coefficient'].append(np.mean(val_dsc, axis=0).tolist())


		e = int(time.time() - start)
		print('-'*100)
		print('Complete')
		print('-'*100)
		train_time = '{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60)
		print('Train time: ', train_time)
		print('='*100)

		return history, train_time



	def report_json(self, history, time):
		log = OrderedDict()
		m_compile = OrderedDict()
		log['ID'] = self.id
		log['NAME'] = self.name
		log['TIME'] = time
		log['INPUT_SHAPE'] = self.input_shape
		log['DATA'] = self.data
		log['CLASS_NUM'] = self.class_num
		log['IMAGE_SIZE'] = self.image_size
		log['CV_RATE'] = self.cv_rate
		log['CV_SEED'] = self.cv_seed
		log['EPOCHS'] = self.epochs
		log['BATCH_SIZE'] =self.batch_size
		log['LEARNINGRATE'] = self.learning_rate
		log['DIMENSION'] = self.dimension

		log['OPTIMIZER'] = self.optimizer
		log['LOSS'] = self.loss
		log['METRICS'] = 'Dice_Coefficent'
#		log['COMPILE'] = m_compile

		# h = history.params
		log['LOSS'] = history['loss']
		log['DSC_TOTAL'] = history['dice_coefficient']
		log['VAL_LOSS'] = history['val_loss']
		log['VAL_DSC_TOTAL'] = history['val_dice_coefficient']
		# if self.input_channel > 1:
		# 	h['DSC0'] = history.history['DSC_0']
		# 	h['DSC1'] = history.history['DSC_1']
		# 	h['DSC2'] = history.history['DSC_2']
		# 	h['DSC3'] = history.history['DSC_3']
		# 	h['DSC4'] = history.history['DSC_4']
		# 	h['DSC5'] = history.history['DSC_5']
		# 	h['DSC6'] = history.history['DSC_6']
		# 	h['DSC7'] = history.history['DSC_7']
		# 	h['VAL_DSC0'] = history.history['val_DSC_0']
		# 	h['VAL_DSC1'] = history.history['val_DSC_1']
		# 	h['VAL_DSC2'] = history.history['val_DSC_2']
		# 	h['VAL_DSC3'] = history.history['val_DSC_3']
		# 	h['VAL_DSC4'] = history.history['val_DSC_4']
		# 	h['VAL_DSC5'] = history.history['val_DSC_5']
		# 	h['VAL_DSC6'] = history.history['val_DSC_6']
		# 	h['VAL_DSC7'] = history.history['val_DSC_7']
		log['HISTORY'] = history


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