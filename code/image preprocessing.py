import nibabel as nib
import numpy as np
import os

from PIL import Image
from tqdm import tqdm
import h5py

# image directory path
root = os.path.abspath(os.path.join(os.getcwd(),"../dataset/"))

ct_set = os.path.join(root, "ct_train_test/ct_train/")
mr_set = os.path.join(root, "mr_train_test/mr_train/")

ct_test_dir = os.path.join(root,"ct_train_test/ct_test/")
mr_test_dir = os.path.join(root,'mr_train_test/mr_test/')

save_dir = os.path.join(root, 'h5py')

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


def resizing_3d(images):
    processed_imgs = list()
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

        # pad_img = nib.Nifti1Image(pad_img, affine=np.eye(4))
        pad_img = pad_img.reshape(256,256,256,1)
        processed_imgs.append(pad_img)

    return np.array(processed_imgs)


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



# image preprocessing
print('=' * 100)
print('Training')
print('-'*100)
print('MR processing')
print('-'*100)
ct_pad_images = resizing_3d(ct_images)
ct_pad_labels = resizing_3d(ct_labels)

print('CT processing')
print('-' * 100)
mr_pad_images = resizing_3d(mr_images)
mr_pad_labels = resizing_3d(mr_labels)



print('=' * 100)
print('Test')
print('-'*100)
print('MR processing')
print('-'*100)
ct_test_pad_images = resizing_3d(ct_test_images)

print('CT processing')
print('-' * 100)
mr_test_pad_images = resizing_3d(mr_test_images)



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