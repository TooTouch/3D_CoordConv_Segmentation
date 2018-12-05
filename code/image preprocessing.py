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


def image_preprocess(images):
    '''
    preprocessing image : zero padding, rescale, onehotencoding
    :param images: Raw images
    :return: preprocessed images
    '''
    processed_imgs = np.zeros((len(images),256,256,256,1))
    for img in tqdm(images):
        resized_img = np.zeros((256, 256, img.shape[2]))
        pad_img = np.zeros((256, 256, 256))
        for i in range(img.shape[2]):
            im = Image.fromarray(img.get_data()[:, :, i])
            resized_img[:, :, i] = np.asarray(im.resize((256, 256)))

        resized_img[resized_img < 0] = 0

        for j in range(resized_img.shape[0]):
            im = Image.fromarray(resized_img[j, :, :])
            im = resizing(im, 256)
            pad_img[j, :, :] = np.asarray(im)

        pad_img = pad_img.reshape(1,256,256,256,1)
        pad_img = pad_img / 255.

        processed_imgs = np.vstack([processed_imgs, pad_img])

    return processed_imgs


def mask_preprocess(masks, class_num=7):
    class_num = class_num + 1  # background
    onehot_masks = np.zeros((len(masks), 256, 256, 256, class_num))

    for mask in tqdm(masks):
        initial_data = mask.get_data()

        initial_size_x = initial_data.shape[0]
        initial_size_y = initial_data.shape[1]
        initial_size_z = initial_data.shape[2]

        new_size_x = 256
        new_size_y = 256
        new_size_z = 256

        delta_x = initial_size_x / new_size_x
        delta_y = initial_size_y / new_size_y
        delta_z = initial_size_z / new_size_z

        new_data = np.zeros((new_size_x, new_size_y, new_size_z))

        for x, y, z in itertools.product(range(new_size_x),
                                         range(new_size_y),
                                         range(new_size_z)):
            new_data[x][y][z] = initial_data[int(x * delta_x)][int(y * delta_y)][int(z * delta_z)]

        # encoder
        raw_shape = new_data.shape
        flatten_data = new_data.reshape(-1)
        label_encoder = preprocessing.LabelEncoder()
        label_data = label_encoder.fit_transform(flatten_data)
        onehot_data = to_categorical(label_data, class_num)
        # reshape to raw shape
        onehot_data = onehot_data.reshape((1,) + raw_shape + (class_num,))
        onehot_masks = np.vstack([onehot_masks, onehot_data])

    return onehot_masks


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
ct_pad_images = image_preprocess(ct_images); del(ct_images)
ct_pad_labels = mask_preprocess(ct_labels); del(ct_labels)


print('MR processing')
print('-' * 100)
mr_pad_images = image_preprocess(mr_images); del(mr_images)
mr_pad_labels = mask_preprocess(mr_labels); del(mr_labels)
print('=' * 100)
print('Test')
print('-'*100)
print('CT processing')
print('-'*100)
ct_test_pad_images = image_preprocess(ct_test_images)

print('MR processing')
print('-' * 100)
mr_test_pad_images = mask_preprocess(mr_test_images)



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