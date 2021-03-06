import numpy as np

from utils import *


def trainGenerator(imgs, labels, patients, batch_size, patch_dim, patch_size, output_chn, input_chn=1,
                   coordnet=False, elastic=False):
    '''
    Data generator
    :param imgs: 3D images
    :param labels: 3D labels
    :param patients: patients
    :param batch_size: Number of images to be entered simultaneously
    :param patch_dim: Patch dimension
    :param patch_size: Number of patch images
    :param output_chn: output channel
    :return: batch_images, batch_labels
    '''
    while True:
        np.random.shuffle(patients)
        img, label = imgs[patients[0]], labels[patients[0]]
        if coordnet:
            img = CoordinateChannel3D(img)
        if elastic:
            im_merge = np.concatenate([img[..., None]] + [label[..., None]], axis=3)
            img, label = elastic_transform3D_fixed(im_merge, im_merge.shape[1] * 2, im_merge.shape[1] *0.08, min(im_merge.shape[1:-1])*0.01)


        for i in range(patch_size):
            batch_imgs, batch_labels = get_batch_patches(img, label, patch_dim, output_chn, batch_size, input_chn)
            yield batch_imgs, batch_labels

def validGenerator(imgs, labels, patients, batch_size, patch_dim, patch_size, output_chn, input_chn=1, coordnet=False, elastic=False):
    '''
    Data generator
    :param imgs: 3D images
    :param labels: 3D labels
    :param patients: patients
    :param batch_size: Number of images to be entered simultaneously
    :param patch_dim: Patch dimension
    :param patch_size: Number of patch images
    :param output_chn: output channel
    :return: batch_images, batch_labels
    '''
    while True:
        for patient in patients:
            img, label = imgs[patient], labels[patient]
            if coordnet:
                img = CoordinateChannel3D(img)

            if elastic:
                im_merge = np.concatenate([img[..., None]] + [label[..., None]], axis=3)
                img, label = elastic_transform3D_fixed(im_merge, im_merge.shape[1] * 2, im_merge.shape[1] * 0.08,
                                                       min(im_merge.shape[1:-1]) * 0.01)
            for i in range(patch_size):
                batch_imgs, batch_labels = get_batch_patches(img, label, patch_dim, output_chn, batch_size, input_chn)
                yield batch_imgs, batch_labels
