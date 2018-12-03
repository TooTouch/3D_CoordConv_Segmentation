import h5py
import os
import numpy as np

class Load_Data:
    def __init__(self):
        self.root = os.path.abspath(os.path.join(os.getcwd(),'../dataset/h5py'))
        self.train_dir = os.path.join(self.root, '../h5py/train_hf')
        self.test_dir = os.path.join(self.root, '../h5py.test_hf')

    def train_load(self):
        print('='*100)
        print('Load train data')
        print('Train data directory: ',self.train_dir)
        train_hf = h5py.File(self.train_dir, 'r')

        ct_images = np.array(train_hf['ct_images'].get_data())
        ct_labels = np.array(train_hf['ct_labels'].get_data())
        mr_images = np.array(train_hf['mr_images'].get_data())
        mr_labels = np.array(train_hf['mr_labels'].get_data())

        train_hf.close()
        print('Complete')
        return (ct_images, ct_labels), (mr_images, mr_labels)


    def test_load(self):
        print('='*100)
        print('Load test data')
        print('Test data directory: ',self.test_dir)
        test_hf = h5py.File(self.test_dir, 'r')

        ct_images = np.array(test_hf['ct_test_images'].get_data())
        mr_images = np.array(test_hf['mr_test_images'].get_data())

        test_hf.close()
        print('Complete')
        return ct_images, mr_images


