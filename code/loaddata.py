import h5py
import os
import numpy as np

class Load_Data:
    def __init__(self):
        self.root = os.path.abspath(os.path.join(os.getcwd(),'../dataset/'))

        # self.ct_train_dir = os.path.join(self.root, 'ct_train_test(processed)/ct_train')
        # self.ct_test_dir = os.path.join(self.root, 'ct_train_test(processed)/ct_test')
        # self.mr_train_dir = os.path.join(self.root, 'mr_train_test(processed)/mr_train')
        # self.mr_test_dir = os.path.join(self.root, 'mr_train_test(processed)/mr_test')

        self.train_dir = os.path.join(self.root, 'h5py/train_hf')
        self.test_dir = os.path.join(self.root, 'h5py.test_hf')

    def train_load(self):
        print('='*100)
        print('Load train data')
        print('Train data directory: ',self.train_dir)
        train_hf = h5py.File(self.train_dir, 'r')

        ct_images = np.zeros((20, 128, 128, 128, 1))
        ct_labels = np.zeros((20, 128, 128, 128, 8))
        mr_images = np.zeros((20, 128, 128, 128, 1))
        mr_labels = np.zeros((20, 128, 128, 128, 8))

        for i in range(20):
            ct_images[i,:,:,:,:] = np.array(train_hf['ct_image_{}'.format(i)])
            ct_labels[i,:,:,:,:] = np.array(train_hf['ct_label_{}'.format(i)])
            mr_images[i,:,:,:,:] = np.array(train_hf['mr_image_{}'.format(i)])
            mr_labels[i,:,:,:,:] = np.array(train_hf['mr_label_{}'.format(i)])

        train_hf.close()

        ct_images = np.moveaxis(ct_images,-1,1)
        ct_labels = np.moveaxis(ct_labels,-1,1)
        mr_images = np.moveaxis(mr_images,-1,1)
        mr_labels = np.moveaxis(mr_labels,-1,1)

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


