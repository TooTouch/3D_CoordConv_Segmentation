import h5py
import os
import numpy as np
import cv2
from glob import glob

from utils import *


def data_gen(imgs, labels, patients, batch_size, patch_dim, output_chn):
    while True:
        np.random.shuffle(patients)
        batch_imgs, batch_labels = get_batch_patches(imgs[patients[0]], labels[patients[0]], patch_dim, output_chn, batch_size)
        yield batch_imgs, batch_labels


