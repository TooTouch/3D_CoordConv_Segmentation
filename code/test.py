import pandas as pd 
import numpy as np 
import os
import h5py
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

import argparse


class State_Test:
	def __init__(self, name):
		self.name = name
		self.root_dir = os.path.abspath(os.path.join(os.getcwd(),'../../'))
		self.model_dir = os.path.join(self.root_dir, 'model')
		self.test_idx = os.listdir(os.path.join(self.root_dir,'dataset/test'))
		self.test_dir = os.path.join(self.root_dir, 'dataset/h5py/test')
		self.test_dir += str((224, 224, 3))
		self.submit_dir = os.path.join(self.root_dir, 'result')
		self.c_class = os.listdir(os.path.join(self.root_dir, 'dataset/train'))
		self.hf = h5py.File(self.test_dir,'r')
		self.model = None
	
	def run(self):
		self.model_()

		for i in range(10):
			# load images
			test_gen, batch_size = self.load_data(i)
			# predict
			prob = self.predict_(test_gen, batch_size=batch_size)
			# save to submit file 
			prob_df = pd.DataFrame(prob, columns=self.c_class)
			submit = pd.concat([pd.DataFrame({'img':self.test_idx[:batch_size]}), prob_df], axis=1)
			submit.to_csv(self.submit_dir + '/submit.' + str(i) + '.csv', index=False)
			# drop 
			self.test_idx = self.test_idx[batch_size:]
			del(test_gen); del(prob); del(prob_df); del(submit)

		self.submit_concat()
		self.hf.close()


	def load_data(self, i):
		print('='*50)
		print('Load test {} iamges'.format(str(i)))
		test_imgs = self.hf['test' + str(i)]
		batch_size = test_imgs.shape[0]
		print('Complete')
		print('='*50)
		test_gen = self.image_gen(images=test_imgs, labels=None, batch_size=1, seed=1223)
		del(test_imgs)
		return test_gen, batch_size


	def image_gen(self, images, labels, batch_size, seed=None, shuffle=False):
		datagen = ImageDataGenerator(rescale = 1./255)
		iterator = datagen.flow(
			x = images,
			y = labels,
			batch_size = batch_size,
			seed=seed,
			shuffle=shuffle
		)
		for batch_x in iterator:
			yield batch_x

	def model_(self):
		print('='*50)
		print('Load model')
		self.model = load_model(self.model_dir + '/' + self.name + '.h5')
		print('Complete')
		print('='*50)

	def predict_(self, x_test, batch_size):
		print('='*50)
		print('Start to predict label')
		prob = self.model.predict_generator(x_test, verbose=1, steps=batch_size)
		print('Complete')
		print('='*50)
		return prob

	def submit_concat(self):
		print('Concatenate submit batch files')
		submit_1_10 = os.listdir(self.submit_dir)
		concat_dir = os.listdir(self.submit_dir + '/concat_file')
		ID = str(len(concat_dir) + 1)
		submit_1_10 = [i for i in submit_1_10 if 'submit' in i]
		print(submit_1_10)
		submit_df = list()
		for df in submit_1_10:
			submit_df.append(pd.read_csv(self.submit_dir + '/' + df))
		submit = pd.concat(submit_df, axis=0)

		submit.to_csv(self.submit_dir + '/concat_file/' + ID + '.submit.csv', index=False)
		print('Complete')

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--name', type=str, help='Model name')
	args = parser.parse_args()

	t = State_Test(args.name)
	t.run()


