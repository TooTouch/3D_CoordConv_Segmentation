import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

from keras.utils import to_categorical
from sklearn import preprocessing

from PIL import Image
import itertools
from tqdm import tqdm

import numpy as np
import skimage.io as io
import tensorflow as tf

from tqdm import tqdm
from dltk.io.preprocessing import *
import scipy.ndimage


import pickle

root = 'D:\segmentiation/'
ct_set = os.path.join(root,'ct_train_test/ct_train/')
mr_set = os.path.join(root,'mr_train_test/mr_train/')
filenames = os.listdir(ct_set)


def pickle_iter(path):
    f = open(path, 'rb')
    unpickler = pickle.Unpickler(f)
    try:
        for i in range(9999999999):
            yield unpickler.load()
    except:
        f.close()
        print('pickle generator created')


def pad3d(array):
    height = array.shape[0]
    depth = array.shape[2]

    if (height - depth) % 2:
        pad_front = int((height + 1 - depth) / 2)
        pad_back = int((height - 1 - depth) / 2)
    else:
        pad_front = pad_back = int((height - depth) / 2)

    npad = ((0, 0), (0, 0), (pad_front, pad_back))
    array_padding = np.pad(array, npad, 'constant', constant_values=(0))
    array_padding[array_padding < 0] = 0

    return array_padding


def image_preprocess(image, new_size, mask=False):
    assert np.sum(image.shape == image.shape[0]) != 3

    ratio = new_size / image.shape[0]

    image = scipy.ndimage.zoom(image, zoom=ratio, order=0)

    if mask:
        channel = 7 + 1  # background
        image = image.reshape(-1)

        image = label_encoder.fit_transform(image)

        print("annotation shape", image.shape)
        print("unique value", np.unique(image))

        image = to_categorical(image, class_num)
        print("processed mask shape", image.shape)


    else:
        channel = 1
    # reshape to raw shape
    image = image.reshape((new_size,) * 3 + (channel,))
    print("Reshaped shape", image.shape)

    return image


def nii_loader(path, file_name):
    file_path = os.path.join(path, file_name)
    fn = os.listdir(file_path)
    image = (nib.load(file_path + '/' + fn[0]))
    return image


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def write_image_annotation_pairs_to_tfrecord_from_gen(generator, tfrecords_filename, test=False):
    """Writes given image/annotation pairs to the tfrecords file.
    The function reads each image/anno'tation pair given filenames
    of image and respective annotation and writes it to the tfrecord
    file.
    Parameters
    ----------
    filename_pairs : array of tuples (img_filepath, annotation_filepath)
        Array of tuples of image/annotation filenames
    tfrecords_filename : string
        Tfrecords filename to write the image/annotation pairs
    """
    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    writer = tf.python_io.TFRecordWriter(tfrecords_filename, options=options)

    print("Start writing tfrecord")

    # for img_path, annotation_path in filename_pairs:

    for i, record in enumerate(generator):
        print(i)

        # img = np.array(Image.open(img_path))

        # annotation = np.array(Image.open(annotation_path))

        # Unomment this one when working with surgical data
        # annotation = annotation[:, :, 0]

        # The reason to store image sizes was demonstrated
        # in the previous example -- we have to know sizes
        # of images to later read raw serialized string,

        # convert to 1d array and convert to respective
        # shape that image used to have.
        # height = img.shape[0]

        img = record[0]
        if test:
            pass
        else:
            annotation = record[1]

        height = img.shape[0]

        # width = img.shape[1]
        width = img.shape[1]

        # add depth
        depth = img.shape[2]
        # print(depth)
        # print(img[0].shape) #288,288,140

        # img_raw = img.tostring()
        img_raw = img.tostring()

        ## 여기서 에러
        print("img shape: {}, img_raw shape: {}".format(img.shape, len(img_raw)))

        ## 여기서 에러

        if test == False:
            annotation_raw = annotation.tostring()
            print("annotation_ shape: {}, annotation_raw shape: {}".format(annotation.shape, len(annotation_raw)))

        # print(annotation[1].shape)
        # print(annotation[2].shape)

        # print(annotation_raw)
        # annotation_raw = annotation.tostring()

        if test == False:
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(height),
                'width': _int64_feature(width),
                'depth': _int64_feature(depth),
                'image_raw': _bytes_feature(img_raw),
                'mask_raw': _bytes_feature(annotation_raw)}))
        else:
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(height),
                'width': _int64_feature(width),
                'depth': _int64_feature(depth),
                'image_raw': _bytes_feature(img_raw)
            }))

        writer.write(example.SerializeToString())

    writer.close()


def write_image_annotation_pairs_to_tfrecord_from_listitr(pickle_itr_list, tfrecords_filename):
    """Writes given image/annotation pairs to the tfrecords file.
    The function reads each image/anno'tation pair given filenames
    of image and respective annotation and writes it to the tfrecord
    file.
    Parameters
    ----------
    filename_pairs : array of tuples (img_filepath, annotation_filepath)
        Array of tuples of image/annotation filenames
    tfrecords_filename : string
        Tfrecords filename to write the image/annotation pairs
    """

    print("Start writing tfrecord")
    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    writer = tf.python_io.TFRecordWriter(tfrecords_filename, options=options)

    # for img_path, annotation_path in filename_pairs:

    for pickle_itr in pickle_itr_list:

        for i, record in enumerate(pickle_itr):
            print(i)
            print(record.keys())

            img = record['image']
            annotation = record['label']

            # img = np.array(Image.open(img_path))

            # annotation = np.array(Image.open(annotation_path))

            # Unomment this one when working with surgical data
            # annotation = annotation[:, :, 0]

            # The reason to store image sizes was demonstrated
            # in the previous example -- we have to know sizes
            # of images to later read raw serialized string,
            # convert to 1d array and convert to respective
            # shape that image used to have.
            # height = img.shape[0]

            print(img.shape)
            ##### img = img.reshape()

            height = img.shape[0]

            # width = img.shape[1]
            width = img.shape[1]

            # add depth
            depth = img.shape[2]
            # print(depth)
            # print(img[0].shape) #288,288,140

            # img_raw = img.tostring()
            img_raw = img.tostring()

            ## 여기서 에러
            print("img shape: {}, img_raw shape: {}".format(img.shape, len(img_raw)))

            ## 여기서 에러
            annotation_raw = annotation.tostring()
            print("annotation_ shape: {}, annotation_raw shape: {}".format(annotation.shape, len(annotation_raw)))

            # print(annotation[1].shape)
            # print(annotation[2].shape)

            # print(annotation_raw)
            # annotation_raw = annotation.tostring()

            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(height),
                'width': _int64_feature(width),
                'depth': _int64_feature(depth),
                'image_raw': _bytes_feature(img_raw),
                'mask_raw': _bytes_feature(annotation_raw)}))

            writer.write(example.SerializeToString())

    writer.close()


def read_image_annotation_pairs_from_tfrecord_itr_from_compressed(tfrecords_filename):
    """Return image/annotation pairs from the tfrecords file.
    The function reads the tfrecords file and returns image
    and respective annotation matrices pairs.
    Parameters
    ----------
    tfrecords_filename : string
        filename of .tfrecords file to read from

    Returns
    -------
    image_annotation_pairs : array of tuples (img, annotation)
        The image and annotation that were read from the file
    """
    image_annotation_pairs = []

    ## option...
    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)

    record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename, options=options)

    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)

        height = int(example.features.feature['height']
                     .int64_list
                     .value[0])

        width = int(example.features.feature['width']
                    .int64_list
                    .value[0])

        depth = int(example.features.feature['depth']
                    .int64_list
                    .value[0])

        img_string = (example.features.feature['image_raw']
            .bytes_list
            .value[0])

        annotation_string = (example.features.feature['mask_raw']
            .bytes_list
            .value[0])

        # img_1d = np.fromstring(img_string, dtype=np.uint8)
        img_1d = np.fromstring(img_string)

        print("img_string shape", img_1d.shape)
        print("image :  {}, annotation length : {}".format(len(img_string), len(annotation_string)))

        #        img = img_1d.reshape((height, width, depth, -1))

        img = img_1d.reshape((height, width, depth, 1))

        annotation_1d = np.fromstring(annotation_string, dtype=np.float32)
        #        annotation_1d = np.fromstring(annotation_string)

        # Annotations don't have depth (3rd dimension)
        # TODO: check if it works for other datasets
        #  annotation = annotation_1d.reshape((height, width, depth,-1))

        annotation = annotation_1d.reshape((height, width, depth, 8))

        yield img, annotation


def pre_process_tfrecord_write_itr(path, itr, new_size, test=False):
    for record in tqdm(itr):
        file_name = record['name']
        img_fname = record['image']

        print(path, img_fname)

        ## nii 데이터 불러오기
        img = nii_loader(path, img_fname)
        if test:
            pass
        else:
            label_fname = record['label']
            lab = nii_loader(path, label_fname)

        print("loaded Image Shape", img.get_data().shape,)

        ## 이미지 padding + resizing
        img = pad3d(img.get_data())
        img = image_preprocess(img, new_size=new_size)

        ## 레이블 padding + resizing
        if test:
            pass
        else:

            lab = pad3d(lab.get_data())
            lab = image_preprocess(lab, new_size=new_size, mask=True)

            print("Mask shape when preproceesing ends", lab.shape)

            yield img, lab

        yield img