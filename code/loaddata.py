import h5py
import os
import numpy as np
import cv2
from glob import glob

from utils import *


def data_gen(dir, subjects, image_num, batch_size, patch_dim, resize_r, output_chn):
    pair_list = glob('{}/*.nii/*.nii'.format(dir))
    rename_map = [0, 205, 420, 500, 550, 600, 820, 850]

    # output : resized images, resized labels
    imgs, labels = load_data_pairs(pair_list, resize_r, output_chn, rename_map)
    while True:
        batch_imgs, batch_labels = get_batch_patches(imgs, labels, patch_dim, output_chn, batch_size)

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
