import h5py
import os
import numpy as np

class Load_Data:
    def __init__(self, dataset, class_num, channel, dimension, size):
        '''
        Initial values
        :param dataset: CT or MR
        :param channel: Number of class
        :param dimension: 2D or 3D
        '''
        self.root = os.path.abspath(os.path.join(os.getcwd(),'../dataset/'))

        self.size = size
        self.dataset = dataset
        self.channel = channel
        self.class_num = class_num
        self.dimension = dimension

        self.train_dir = os.path.join(self.root, 'h5py/train_hf' + str(self.size) + '_' + str(self.class_num))


    def test_load(self):
        print('='*100)
        print('Load test data')
        print('Test data directory: ',self.test_dir)
        test_hf = h5py.File(self.test_dir, 'r')

        images = np.zeros((40, self.size, self.size, self.size, 1))
        labels = np.zeros((40, self.size, self.size, self.size, self.channel))

        for i in range(40):
            images[i] = np.array(test_hf['{}_test_image_{}'.format(self.dataset.lower(), i)])
            labels[i] = np.array(test_hf['{}_test_label_{}'.format(self.dataset.lower(), i)])
        test_hf.close()

        if self.dimension == '2D':
            images = images.reshape(-1, self.size, self.size, 1)
            labels = labels.reshape(-1, self.size, self.size, self.channel)

        images = np.moveaxis(images, -1, 1)
        labels = np.moveaxis(labels, -1, 1)

        print('Complete')

        return (images, labels)


    def data_gen(self, batch_size, subjects):
        """
        Generator to yield inputs and their labels in batches.
        """
        d = 2 if self.dimension=='2D' else 3
        train_hf = h5py.File(self.train_dir, 'r')

        while True:
            batch_imgs = np.array([]).reshape((0,) + (1,) + (self.size,) * d )
            batch_labels = np.array([]).reshape((0,) + (self.channel,) + (self.size,) * d)

            for i in range(batch_size):
                idx = np.random.choice(subjects)
                img = np.array(train_hf['{}_image_{}'.format(self.dataset.lower(),idx)]).reshape((-1,) + (self.size,) * d + (1,))
                label = np.array(train_hf['{}_label_split_{}'.format(self.dataset.lower(),idx)])[2].reshape((-1,) + (self.size,) * d + (self.channel,))

                img /= 255.

                img = np.moveaxis(img, -1, 1)
                label = np.moveaxis(label, -1, 1)

                batch_imgs = np.vstack([batch_imgs, img])
                batch_labels = np.vstack([batch_labels, label])

                yield batch_imgs, batch_labels