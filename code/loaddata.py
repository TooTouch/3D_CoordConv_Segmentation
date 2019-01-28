import h5py
import os
import numpy as np
import cv2
from glob import glob

from utils import *


# def test_load(self):
#     print('='*100)
#     print('Load test data')
#     print('Test data directory: ',self.test_dir)
#     test_hf = h5py.File(self.test_dir, 'r')
#
#     images = np.zeros((40, self.size, self.size, self.size, 1))
#     labels = np.zeros((40, self.size, self.size, self.size, self.channel))
#
#     for i in range(40):
#         images[i] = np.array(test_hf['{}_test_image_{}'.format(self.dataset.lower(), i)])
#         labels[i] = np.array(test_hf['{}_test_label_{}'.format(self.dataset.lower(), i)])
#     test_hf.close()
#
#     images = np.moveaxis(images, -1, 1)
#     labels = np.moveaxis(labels, -1, 1)
#
#     print('Complete')
#
#     return (images, labels)

#
# def data_gen(self, batch_size, subjects):
#     """
#     Generator to yield inputs and their labels in batches.
#     """
#     train_hf = h5py.File(self.train_dir, 'r')
#     while True:
#         # batch_imgs = np.array([]).reshape((0,) + (self.size,) * 3  + (1,))
#         # batch_labels = np.array([]).reshape((0,) + (self.size,) * 3 + (self.channel,))
#
#         for i in range(batch_size):
#             idx = np.random.choice(subjects)
#             # img = np.array(train_hf['{}_image_{}'.format(self.dataset.lower(),idx)]).reshape((1,) + (self.size,) * 3 + (1,))
#             # label = np.array(train_hf['{}_label_{}'.format(self.dataset.lower(),idx)]).reshape((1,) + (self.size,) * 3 + (self.channel,))
#
#             img = np.array(train_hf['{}_image_{}'.format(self.dataset.lower(), idx)]).reshape((1,) + (128,) * 3 + (1,))
#             label = np.array(train_hf['{}_label_{}'.format(self.dataset.lower(), idx)]).reshape((1,) + (128,) * 3 + (self.channel,))
#
#             # img = np.moveaxis(img, -1, 1)
#             # label = np.moveaxis(label, -1, 1)
#
#             batch_imgs, batch_labels = self.get_batch_patches(img, label, 96, 2)
#
#             # batch_imgs = np.vstack([batch_imgs, img])
#             # batch_labels = np.vstack([batch_labels, label])
#
#
#             yield batch_imgs, batch_labels



def data_gen(dir, batch_size, patch_dim, resize_r, subjects):
    pair_list = glob('{}/*.nii/*.nii'.format(dir))
    rename_map = [0, 205, 420, 500, 550, 600, 820, 850]

    # output : resized images, resized labels
    while True:
        for i in range(batch_size):
            idx = np.random.choice(subjects)
            imgs, labels = load_data_pairs(pair_list[idx*2:idx*2+2], resize_r, rename_map)
            batch_imgs, batch_labels = get_batch_patches(imgs, labels, patch_dim, batch_size)

            yield batch_imgs, batch_labels




def clahe(img):
    img = img.reshape((128,128,128))
    img_clahe = np.zeros((128, 128, 128))
    for j in range(img.shape[0]):
        cv2.imwrite('img.jpg', img[j, :, :])
        img_j = cv2.imread('img.jpg', cv2.IMREAD_GRAYSCALE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe[j, :, :] = clahe.apply(img_j)
    img_clahe = img_clahe.reshape((-1,) + img_clahe.shape + (1,))

    return img_clahe
