import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import os
import h5py
import numpy as np


class CardiacDataset(Dataset):

    def __init__(self, dataset, class_num, channel, dimension, size, subjects=None):

        self.root = r'D:\data\segmentiation'
        self.size = size
        self.dataset = dataset
        self.channel = channel
        self.class_num = class_num
        self.dimension = dimension
        self.subjects = subjects

        self.train_dir = os.path.join(self.root, 'h5py/train_hf' + str(self.size) + '_' + str(self.class_num))
        print("Train Dataset Directory", self.train_dir)

        train_hf = h5py.File(self.train_dir, 'r')

        images = np.zeros((20, self.size, self.size, self.size, 1))
        labels = np.zeros((20, self.size, self.size, self.size, self.channel))

        if subjects:

            self.len = len(subjects)

            images = np.zeros((len(subjects), self.size, self.size, self.size, 1))
            labels = np.zeros((len(subjects), self.size, self.size, self.size, self.channel))

            for i, j in enumerate(subjects):
                images[i] = np.array(train_hf['{}_image_{}'.format(self.dataset.lower(), j)])
                labels[i] = np.array(train_hf['{}_label_{}'.format(self.dataset.lower(), j)])

        else:

            self.len = 20

            images = np.zeros((20, self.size, self.size, self.size, 1))
            labels = np.zeros((20, self.size, self.size, self.size, self.channel))

            for i in range(20):
                #print("Loading Image", i)
                images[i] = np.array(train_hf['{}_image_{}'.format(self.dataset.lower(), i)])
                labels[i] = np.array(train_hf['{}_label_{}'.format(self.dataset.lower(), i)])

        train_hf.close()

        self.x_data = torch.from_numpy(images)
        self.y_data = torch.from_numpy(labels)

    #         xy = np.loadtxt('./data/diabetes.csv.gz',
    #                         delimiter=',', dtype=np.float32)
    #         self.len = xy.shape[0]
    #         self.x_data = torch.from_numpy(xy[:, 0:-1])
    #         self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        #print("Getting image", index)
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len
