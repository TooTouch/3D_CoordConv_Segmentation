import nibabel as nib
import numpy as np
import os

from PIL import Image

from keras.utils import to_categorical
from sklearn import preprocessing

from tqdm import tqdm
import h5py

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


# Functions
def resizing(img, resize):
    bigsize = max(img.size)
    bg = Image.new(mode='L', size=(bigsize,bigsize), color=0)
    offset = (int(round(((bigsize - img.size[0])/2),0)), int(round(((bigsize - img.size[1])/2),0)))
    bg.paste(img,offset)
    bg = bg.resize((resize,resize))

    return bg

def onehotencoder(img, class_num):
    '''
    :param img: zero padding image
    :param class_num: Number of class and background
    :return: converted to one-hot-vector by voxel (256,256,256,8)
    '''

    # flatten
    raw_shape = img.shape
    flatten_img = img.reshape(-1)
    class_num = class_num + 1 # background
    # encoder
    label_encoder = preprocessing.LabelEncoder()
    label_img = label_encoder.fit_transform(flatten_img)
    onehot_img = to_categorical(label_img, class_num)
    # reshape to raw shape
    onehot_img = onehot_img.reshape(raw_shape[:-1] + (class_num,))

    return onehot_img

def preprocess(images, mask=False):
    '''
    preprocess image : zero padding, rescale, onehotencoding
    :param images: Raw images
    :param mask: If mask is True, then class values convert to one-hot-vector
    :return: preprocessed images
    '''
    processed_imgs = list()
    for img in tqdm(images):
        resized_img = np.zeros((128, 128, img.shape[2]))
        pad_img = np.zeros((128, 128, 128))
        for i in range(img.shape[2]):
            im = Image.fromarray(img.get_data()[:, :, i])
            resized_img[:, :, i] = np.asarray(im.resize((128, 128)))

        resized_img[resized_img < 0] = 0

        for j in range(resized_img.shape[0]):
            im = Image.fromarray(resized_img[j, :, :])
            im = resizing(im, 128)
            pad_img[j, :, :] = np.asarray(im)

        pad_img = pad_img.reshape(128,128,128,1)
        pad_img = pad_img / np.max(pad_img)
        # pad_img = nib.Nifti1Image(pad_img, affine=np.eye(4))

        if mask:
            pad_img = onehotencoder(pad_img, class_num=7)

        processed_imgs.append(pad_img)

    return processed_imgs


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
ct_pad_images = preprocess(ct_images); del(ct_images)
ct_pad_labels = preprocess(ct_labels, mask=True); del(ct_labels)
# # Save
# image_idx = 0
# label_idx = 0
# for i in range(len(ct_list)):
#     save_dir = os.path.join(ct_train_save_dir, ct_list[i])
#     if 'image' in ct_list[i]:
#         nib.save(ct_pad_images[image_idx], save_dir)
#         image_idx+=1
#     else:
#         nib.save(ct_pad_labels[label_idx], save_dir)
#         label_idx+=1
# del(ct_pad_images)

print('MR processing')
print('-' * 100)
mr_pad_images = preprocess(mr_images); del(mr_images)
mr_pad_labels = preprocess(mr_labels, mask=True); del(mr_labels)
# # Save
# image_idx = 0
# label_idx = 0
# for i in range(len(mr_list)):
#     save_dir = os.path.join(mr_train_save_dir, mr_list[i])
#     if 'image' in mr_list[i]:
#         nib.save(mr_pad_images[image_idx], save_dir)
#         image_idx += 1
#     else:
#         nib.save(mr_pad_labels[label_idx], save_dir)
#         label_idx += 1
# del(mr_pad_images)

print('=' * 100)
print('Test')
print('-'*100)
print('CT processing')
print('-'*100)
ct_test_pad_images = preprocess(ct_test_images); del(ct_test_images)
# for i in range(len(ct_test_list)):
#     save_dir = os.path.join(ct_test_save_dir, ct_test_list[i])
#     nib.save(ct_test_pad_images[i], save_dir)
# del(ct_test_pad_images)

print('MR processing')
print('-' * 100)
mr_test_pad_images = preprocess(mr_test_images); del(mr_test_images)
# for i in range(len(mr_test_list)):
#     save_dir = os.path.join(mr_test_save_dir, mr_test_list[i])
#     nib.save(mr_test_pad_images[i], save_dir)
# del(mr_test_pad_images)









with h5py.File(save_dir + '/train_hf','w') as train_set:
    train_set.create_dataset('ct_images', data=ct_pad_images, compression='lzf')
    train_set.create_dataset('ct_labels', data=ct_pad_labels, compression='lzf')
    train_set.create_dataset('mr_images', data=mr_pad_images, compression='lzf')
    train_set.create_dataset('mr_labels', data=mr_pad_labels, compression='lzf')
train_set.close()


with h5py.File(save_dir + '/test_hf','w') as test_set:
    test_set.create_dataset('ct_test_images', data=ct_test_pad_images, compression='lzf')
    test_set.create_dataset('mr_test_images', data=mr_test_pad_images, compression='lzf')
test_set.close()