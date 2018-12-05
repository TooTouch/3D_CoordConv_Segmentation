import nibabel as nib
import numpy as np
import os

from keras.utils import to_categorical
from sklearn import preprocessing

from tqdm import tqdm
from dltk.io.preprocessing import *
import h5py

# Number of class (include background)
class_num = 7 + 1

# image directory path
root = os.path.abspath(os.path.join(os.getcwd(),"../dataset/"))

ct_set = os.path.join(root, "ct_train_test/ct_train/")
mr_set = os.path.join(root, "mr_train_test/mr_train/")

ct_test_dir = os.path.join(root,"ct_train_test/ct_test/")
mr_test_dir = os.path.join(root,'mr_train_test/mr_test/')

save_dir = os.path.join(root,'h5py')

ct_train_save_dir = os.path.join(root, 'ct_train_test(processed)/ct_train')
ct_test_save_dir = os.path.join(root, 'ct_train_test(processed)/ct_test')
mr_train_save_dir = os.path.join(root, 'mr_train_test(processed)/mr_train')
mr_test_save_dir = os.path.join(root, 'mr_train_test(processed)/mr_test')

# list directory
ct_list = os.listdir(ct_set)
mr_list = os.listdir(mr_set)
ct_test_list = os.listdir(ct_test_dir)
mr_test_list = os.listdir(mr_test_dir)

# image list
ct_images = list()
ct_labels = list()
mr_images = list()
mr_labels = list()

ct_test_images = list()
ct_test_pad_images = list()

mr_test_images = list()
mr_test_pad_images = list()

ct_pad_images = list()
ct_pad_labels = list()
mr_pad_images = list()
mr_pad_labels = list()


# hfd5
train_set = h5py.File(save_dir + '/train_hf','w')
test_set = h5py.File(save_dir + '/test_hf','w')


# Load train images
for ct_l in ct_list:
    if 'image' in ct_l:
        file_path = os.path.join(ct_set, ct_l)
        fn = os.listdir(file_path)
        ct_images.append(nib.load(file_path + '/' + fn[0]))
    elif 'label' in ct_l:
        file_path = os.path.join(ct_set, ct_l)
        fn = os.listdir(file_path)
        ct_labels.append(nib.load(file_path + '/' + fn[0]))


for mr_l in mr_list:
    if 'image' in mr_l:
        file_path = os.path.join(mr_set, mr_l)
        fn = os.listdir(file_path)
        mr_images.append(nib.load(file_path + '/' + fn[0]))
    elif 'label' in mr_l:
        file_path = os.path.join(mr_set, mr_l)
        fn = os.listdir(file_path)
        mr_labels.append(nib.load(file_path + '/' + fn[0]))


# Load test images
for fn in ct_test_list:
    img_dir = os.path.join(ct_test_dir, fn)
    img_fn = os.listdir(img_dir)[0]
    im = nib.load(os.path.join(img_dir, img_fn))
    ct_test_images.append(im)


for fn in mr_test_list:
    img_dir = os.path.join(mr_test_dir, fn)
    img_fn = os.listdir(img_dir)[0]
    im = nib.load(os.path.join(img_dir, img_fn))
    mr_test_images.append(im)

# Fix label value
m = mr_labels[9].get_data()
m[m==421] = 420


# Image shape
print('='*100)
print('CT shape')
print('='*100)
for ct_image in ct_images:
    print(ct_image.shape)
print('='*100)
print('MR shape')
print('='*100)
for mr_image in mr_images:
    print(mr_image.shape)

# image preprocess
print('=' * 100)
print('Training')
print('-'*100)
print('CT processing')
print('-'*100)


for i in tqdm(range(len(ct_images))):
    img = ct_images[i].get_data()
    img = resize_image_with_crop_or_pad(img, [128, 128, 128], mode='symmetric').reshape(128,128,128,1)

    train_set.create_dataset('ct_image_{}'.format(i), data=img, compression='lzf')
    del(img)

ct_pad_labels = np.zeros((len(ct_labels), 128, 128, 128, 8))
label_encoder = preprocessing.LabelEncoder()
for i in tqdm(range(len(ct_labels))):
    img = ct_labels[i].get_data()
    img = resize_image_with_crop_or_pad(img, [128, 128, 128], mode='symmetric')

    # encoder
    raw_shape = img.shape
    img = img.reshape(-1)
    img = label_encoder.fit_transform(img)
    img = to_categorical(img, class_num)

    # reshape to raw shape
    img = img.reshape(raw_shape + (class_num,))

    train_set.create_dataset('ct_label_{}'.format(i), data=img, compression='lzf')

print('MR processing')
print('-' * 100)
mr_pad_images = np.zeros((len(mr_images),128,128,128,1))
for i in tqdm(range(len(mr_images))):
    img = mr_images[i].get_data()
    img = resize_image_with_crop_or_pad(img, [128, 128, 128], mode='symmetric').reshape(128,128,128,1)
    train_set.create_dataset('mr_image_{}'.format(i), data=img, compression='lzf')


mr_pad_labels = np.zeros((len(mr_labels), 128, 128, 128, 8))
label_encoder = preprocessing.LabelEncoder()
for i in tqdm(range(len(mr_labels))):
    img = mr_labels[i].get_data()
    img = resize_image_with_crop_or_pad(img, [128, 128, 128], mode='symmetric')

    # encoder
    raw_shape = img.shape
    img = img.reshape(-1)

    img = label_encoder.fit_transform(img)
    label, cnt = np.unique(img, return_counts=True)
    img = to_categorical(img, class_num)

    # reshape to raw shape
    img = img.reshape(raw_shape + (class_num,))
    train_set.create_dataset('mr_label_{}'.format(i), data=img, compression='lzf')
train_set.close()



print('=' * 100)
print('Test')
print('-'*100)
print('CT processing')
print('-'*100)
for i in tqdm(range(len(ct_test_images))):
    img = ct_test_images[i].get_data()
    img = resize_image_with_crop_or_pad(img, [128, 128, 128], mode='symmetric').reshape(128,128,128,1)
    test_set.create_dataset('ct_test_{}'.format(i), data=img, compression='lzf')

print('MR processing')
print('-' * 100)
for i in tqdm(range(len(mr_test_images))):
    img = mr_test_images[i].get_data()
    img = resize_image_with_crop_or_pad(img, [128, 128, 128], mode='symmetric').reshape(128,128,128,1)
    test_set.create_dataset('mr_test_{}'.format(i), data=img, compression='lzf')
test_set.close()




