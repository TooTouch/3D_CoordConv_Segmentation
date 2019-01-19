import nibabel as nib
import numpy as np
import os

from keras.utils import to_categorical
from sklearn import preprocessing
import scipy.ndimage

from tqdm import tqdm
import h5py
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--class_n', type=int, default=7, help='Number of classes')
parser.add_argument('--size', type=int, default=256, help='Image size')
args = parser.parse_args()


# Number of class (include background)
class_num = args.class_n
# Image size
size = args.size

print('='*100)
print('Start image preprocessing')
print('-'*100)
print('Number of class: ',class_num)
print('Image size: ',size)

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

# label encoder
label_encoder = preprocessing.LabelEncoder()

# hfd5
train_set = h5py.File(save_dir + '/train_hf'+str(size) + '_' + str(class_num),'w')
test_set = h5py.File(save_dir + '/test_hf'+str(size),'w')

# functions
def pad3d(array):
    x0 = array.shape[0]
    x2 = array.shape[2]
    if x0 > x2:
        height = x0
        depth = x2
    elif x2 > x0:
        height = x2
        depth = x0

    if (height - depth) % 2:
        pad_front = int((height + 1 - depth) / 2)
        pad_back = int((height - 1 - depth) / 2)
    else:
        pad_front = pad_back = int((height - depth) / 2)

    if x0 > x2:
        npad = ((0, 0), (0, 0), (pad_front, pad_back))
    elif x2 > x0:
        npad = ((pad_front, pad_back), (0, 0), (0, 0))

    array_padding = np.pad(array, npad, 'constant', constant_values=(0))
    array_padding[array_padding < 0] = 0

    return array_padding


def image_preprocess(image, new_size, mask=False, channel=1):
    assert np.sum(image.shape == image.shape[0]) != 3

    ratio = new_size / image.shape[0]

    image = scipy.ndimage.zoom(image, zoom=ratio, order=0)

    if mask:
        if channel > 1:
            channel += 1 # background
            image = image.reshape(-1)
            image = label_encoder.fit_transform(image)
            image = to_categorical(image, channel)
        else:
            image[image>0] = 1

    # reshape to raw shape
    image = image.reshape((new_size,) * 3 + (channel,))

    return image

def axis_transform(image):
    idx = 0
    if np.sum(image[5,:,int(image.shape[-1]/2)]) == 0:
        image = image.T
        image = np.flip(image,axis=0)
        idx += 1
    return image, idx


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
    image = ct_images[i].get_data()
    image_mask = ct_images[i].get_data()
    label = ct_labels[i].get_data()

    # image save
    image = pad3d(image)
    image = image_preprocess(image, new_size=size)
    train_set.create_dataset('ct_image_{}'.format(i), data=image, compression='lzf')

    # masked image save
    mask = label == 0
    image_mask[mask] = 0
    image_mask = pad3d(image_mask)
    image_mask = image_preprocess(image_mask, new_size=size)
    train_set.create_dataset('ct_image_mask_{}'.format(i), data=image_mask, compression='lzf')

    del image, image_mask

    # multiclass label save
    label = pad3d(label)
    label = image_preprocess(label, new_size=size, mask=True, channel=class_num)
    train_set.create_dataset('ct_label_{}'.format(i), data=label, compression='lzf')

    # label by class save
    s_label = np.argmax(label, axis=-1)
    s_label = s_label.reshape(s_label.shape + (1,))

    c = np.zeros((7, size, size, size, 1))
    for j in range(1, 8):
        mask = s_label == j
        c[j - 1, :, :, :, :] = mask.astype(int)
    train_set.create_dataset('ct_label_split_{}'.format(i), data=c, compression='lzf')

    del label, c



print('MR processing')
print('-' * 100)
transform_idx = list()
for i in tqdm(range(len(mr_images))):
    image = mr_images[i].get_data()
    image_mask = mr_images[i].get_data()
    label = mr_labels[i].get_data()

    # image save
    image, idx = axis_transform(image)
    image = pad3d(image)
    image = image_preprocess(image, new_size=size)
    train_set.create_dataset('mr_image_{}'.format(i), data=image, compression='lzf')
    transform_idx.append(idx)

    # masked image save
    mask = label == 0
    image_mask[mask] = 0
    if transform_idx[i]:
        image_mask, _ = axis_transform(image_mask)
    image_mask = pad3d(image_mask)
    image_mask = image_preprocess(image_mask, new_size=size)
    train_set.create_dataset('mr_image_mask_{}'.format(i), data=image_mask, compression='lzf')

    del image, image_mask

    # multiclass label save
    if transform_idx[i]:
        label, _ = axis_transform(label)
    label = pad3d(label)
    label = image_preprocess(label, new_size=size, mask=True, channel=class_num)
    train_set.create_dataset('mr_label_{}'.format(i), data=label, compression='lzf')

    # label by class save
    s_label = np.argmax(label, axis=-1)
    s_label = s_label.reshape(s_label.shape + (1,))
    c = np.zeros((7, size, size, size, 1))
    for j in range(1, 8):
        mask = s_label == j
        c[j - 1, :, :, :, :] = mask.astype(int)
    train_set.create_dataset('mr_label_split_{}'.format(i), data=c, compression='lzf')

    del label, c

train_set.close()



print('=' * 100)
print('Test')
print('-'*100)
print('CT processing')
print('-'*100)
for i in tqdm(range(len(ct_test_images))):
    image = ct_test_images[i].get_data()
    image = pad3d(image)
    image = image_preprocess(image, new_size=size)
    test_set.create_dataset('ct_test_{}'.format(i), data=image, compression='lzf')

print('MR processing')
print('-' * 100)
for i in tqdm(range(len(mr_test_images))):
    image = mr_test_images[i].get_data()
    image = pad3d(image)
    image = image_preprocess(image, new_size=size)
    test_set.create_dataset('mr_test_{}'.format(i), data=image, compression='lzf')

test_set.close()




