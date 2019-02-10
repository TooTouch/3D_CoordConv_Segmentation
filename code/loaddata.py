import h5py
import os
import numpy as np
#import cv2
import copy

class Load_Data:
    def __init__(self, dataset, class_num, channel, dimension, size):
        '''
        Initial values
        :param dataset: CT or MR
        :param channel: Number of class
        :param dimension: 2D or 3D
        '''
        #self.root = os.path.abspath(os.path.join(os.getcwd(),'../dataset/'))
        self.root = r'D:\data\segmentiation'
        self.size = size
        self.dataset = dataset
        self.channel = channel
        self.class_num = class_num
        self.dimension = dimension

        self.train_dir = os.path.join(self.root, 'h5py/train_hf' + str(self.size) + '_' + str(self.class_num))
        print("Train Dataset Directory", self.train_dir)
        #self.train_dir = os.path.join(self.root, 'h5py/train_hf128_' + str(self.class_num))
        # self.train_dir = os.path.join(self.root, 'ct_train_test/ct_train')

    def test_load(self):
        print('='*100)
        print('Load test data')
        print('Test data directory: ',self.test_dir)
        test_hf = h5py.File(self.test_dir, 'r')

        images = np.zeros((40, self.size, self.size, self.size, 1))
        labels = np.zeros((40, self.size, self.size, self.size, self.channel))

        for i in range(40):
            images[i] = np.array(test_hf['{}_test_image_{}'.format(self.dataset.lower(), i)])
            labels[i] = np.array(test_hf['{}_test_label_{}'.format(self.dataset.lower(), i)])
        test_hf.close()

        images = np.moveaxis(images, -1, 1)
        labels = np.moveaxis(labels, -1, 1)

        print('Complete')

        return (images, labels)


    def data_gen(self, batch_size, subjects):
        """
        Generator to yield inputs and their labels in batches.
        """
        print("Train Dataset Directory", self.train_dir)

        train_hf = h5py.File(self.train_dir, 'r')
        while True:
            # batch_imgs = np.array([]).reshape((0,) + (self.size,) * 3  + (1,))
            # batch_labels = np.array([]).reshape((0,) + (self.size,) * 3 + (self.channel,))

            batch_imgs = np.array([]).reshape((0,) + (64,) * 3  + (1,))
            batch_labels = np.array([]).reshape((0,) + (64,) * 3 + (self.channel,))

            for i in range(batch_size):
                idx = np.random.choice(subjects)
                img = np.array(train_hf['{}_image_{}'.format(self.dataset.lower(),idx)]).reshape((1,) + (64,) * 3 + (1,))
                label = np.array(train_hf['{}_label_{}'.format(self.dataset.lower(),idx)]).reshape((1,) + (64,) * 3 + (self.channel,))

                # img = np.array(train_hf['{}_image_{}'.format(self.dataset.lower(), idx)]).reshape((1,) + (self.size,) * 3 + (1,))
                # label = np.array(train_hf['{}_label_{}'.format(self.dataset.lower(), idx)]).reshape((1,) + (self.size,) * 3 + (self.channel,))
                #
                # img = np.moveaxis(img, -1, 1)
                # label = np.moveaxis(label, -1, 1)
                #
                # batch_imgs, batch_labels = self.get_batch_patches(img, label, 96, 2)

                batch_imgs = np.vstack([batch_imgs, img])
                batch_labels = np.vstack([batch_labels, label])


                yield batch_imgs, batch_labels


    def get_batch_patches(self, img_clec, label_clec, patch_dim, batch_size, chn=1, flip_flag=True, rot_flag=True):
        """generate a batch of paired patches for training"""
        batch_img = np.zeros([batch_size, patch_dim, patch_dim, patch_dim, chn]).astype('float32')
        batch_label = np.zeros([batch_size, patch_dim, patch_dim, patch_dim, 8]).astype('int32')


        for k in range(batch_size):
            # randomly select an image pair
            rand_idx = np.arange(len(img_clec))
            np.random.shuffle(rand_idx)
            rand_img = img_clec[rand_idx[0]]
            rand_label = label_clec[rand_idx[0]]
            rand_img = rand_img.astype('float32')
            rand_label = rand_label.astype('int32')

            # randomly select a box anchor
            l, w, h = rand_img.shape[:-1]
            l_rand = np.arange(l - patch_dim)
            w_rand = np.arange(w - patch_dim)
            h_rand = np.arange(h - patch_dim)
            np.random.shuffle(l_rand)
            np.random.shuffle(w_rand)
            np.random.shuffle(h_rand)
            pos = np.array([l_rand[0], w_rand[0], h_rand[0]])
            # crop
            img_temp = copy.deepcopy(
                rand_img[pos[0]:pos[0] + patch_dim, pos[1]:pos[1] + patch_dim, pos[2]:pos[2] + patch_dim, :])
            # normalization
            img_temp = img_temp / 255.0
            mean_temp = np.mean(img_temp)
            dev_temp = np.std(img_temp)
            img_norm = (img_temp - mean_temp) / dev_temp

            label_temp = copy.deepcopy(
                rand_label[pos[0]:pos[0] + patch_dim, pos[1]:pos[1] + patch_dim, pos[2]:pos[2] + patch_dim, :])

            # # possible augmentation
            # # rotation
            # if rot_flag and np.random.random() > 0.65:
            #     # print 'rotating patch...'
            #     rand_angle = [-25, 25]
            #     np.random.shuffle(rand_angle)
            #     img_norm = rotate(img_norm, angle=rand_angle[0], axes=(1, 0), reshape=False, order=1)
            #     label_temp = rotate(label_temp, angle=rand_angle[0], axes=(1, 0), reshape=False, order=0)

            batch_img[k, :, :, :, :] = img_norm
            batch_label[k, :, :, :, :] = label_temp

        return batch_img, batch_label

    def load_data_pairs(pair_list, resize_r, rename_map):
        """load all volume pairs"""
        img_clec = []
        label_clec = []

        # rename_map = [0, 205, 420, 500, 550, 600, 820, 850]
        for k in range(0, len(pair_list), 2):
            img_path = pair_list[k]
            lab_path = pair_list[k + 1]
            img_data = nib.load(img_path).get_data().copy()
            lab_data = nib.load(lab_path).get_data().copy()

            ###preprocessing
            # resize
            resize_dim = (np.array(img_data.shape) * resize_r).astype('int')
            img_data = resize(img_data, resize_dim, order=1, preserve_range=True)
            lab_data = resize(lab_data, resize_dim, order=0, preserve_range=True)
            lab_r_data = np.zeros(lab_data.shape, dtype='int32')

            # rename labels
            for i in range(len(rename_map)):
                lab_r_data[lab_data == rename_map[i]] = i

            # for s in range(img_data.shape[2]):
            #     cv2.imshow('img', np.concatenate(((img_data[:,:,s]).astype('uint8'), (lab_r_data[:,:,s]*30).astype('uint8')), axis=1))
            #     cv2.waitKey(20)

            img_clec.append(img_data)
            label_clec.append(lab_r_data)

        return img_clec, label_clec


    def clahe(self, img):
        img = img.reshape((128,128,128))
        img_clahe = np.zeros((128, 128, 128))
        for j in range(img.shape[0]):
            cv2.imwrite('img.jpg', img[j, :, :])
            img_j = cv2.imread('img.jpg', cv2.IMREAD_GRAYSCALE)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_clahe[j, :, :] = clahe.apply(img_j)
        img_clahe = img_clahe.reshape((-1,) + img_clahe.shape + (1,))

        return img_clahe
