import numpy as np
import copy
import nibabel as nib
from skimage.transform import resize
from keras.utils import to_categorical

import keras.backend as K
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os
import io
from keras.callbacks import TensorBoard

def load_data_pairs(pair_list, resize_r, output_chn, rename_map):
    """load all volume pairs"""
    img_list = []
    label_list = []

    for k in tqdm(range(0, len(pair_list), 2)):
        img_path = pair_list[k]
        lab_path = pair_list[k + 1]
        img_data = nib.load(img_path).get_data().copy()
        lab_data = nib.load(lab_path).get_data().copy()

        ###preprocessing
        # resize
        resize_dim = (np.array(img_data.shape) * resize_r).astype('int')
        img_data = resize(img_data, resize_dim, order=1, preserve_range=True, mode = 'constant')
        lab_data = resize(lab_data, resize_dim, order=0, preserve_range=True, mode = 'constant')
        lab_r_data = np.zeros(lab_data.shape, dtype='int32')

        # rename labels
        for i in range(len(rename_map)):
            lab_r_data[lab_data == rename_map[i]] = i

        lab_r_data = to_categorical(lab_r_data, output_chn)
        img_data = img_data.reshape(img_data.shape + (1,))

        img_list.append(img_data)
        label_list.append(lab_r_data)

    return img_list, label_list



def get_batch_patches(img, label, patch_dim, output_chn, batch_size, chn=1, flip_flag=True, rot_flag=True):
    """generate a batch of paired patches for training"""
    batch_img = np.zeros([batch_size, patch_dim, patch_dim, patch_dim, chn]).astype('float32')
    batch_label = np.zeros([batch_size, patch_dim, patch_dim, patch_dim, output_chn]).astype('int32')

    for k in range(batch_size):
        img = img.astype('float32')
        label = label.astype('int32')

        # randomly select a box anchor
        l, w, h = img.shape[:-1]
        l_rand = np.arange(l - patch_dim)
        w_rand = np.arange(w - patch_dim)
        h_rand = np.arange(h - patch_dim)
        np.random.shuffle(l_rand)
        np.random.shuffle(w_rand)
        np.random.shuffle(h_rand)
        pos = np.array([l_rand[0], w_rand[0], h_rand[0]])
        # crop
        img_temp = copy.deepcopy(
            img[pos[0]:pos[0] + patch_dim, pos[1]:pos[1] + patch_dim, pos[2]:pos[2] + patch_dim, :])
        # normalization
        img_temp = img_temp / 255.0
        # mean_temp = np.mean(img_temp)
        # dev_temp = np.std(img_temp)
        # img_norm = (img_temp - mean_temp) / dev_temp

        label_temp = copy.deepcopy(label[pos[0]:pos[0] + patch_dim, pos[1]:pos[1] + patch_dim, pos[2]:pos[2] + patch_dim, :])

        # # possible augmentation
        # # rotation
        # if rot_flag and np.random.random() > 0.65:
        #     # print 'rotating patch...'
        #     rand_angle = [-25, 25]
        #     np.random.shuffle(rand_angle)
        #     img_norm = rotate(img_norm, angle=rand_angle[0], axes=(1, 0), reshape=False, order=1)
        #     label_temp = rotate(label_temp, angle=rand_angle[0], axes=(1, 0), reshape=False, order=0)

        batch_img[k, :, :, :, :] = img_temp
        batch_label[k, :, :, :, :] = label_temp

    return batch_img, batch_label


def CoordinateChannel3D(image):
    assert len(image.shape) == 4  # weight, height, depth, channel
    batch_shape = 1
    input_shape = image.shape
    input_shape = [input_shape[i] for i in range(4)]
    dim1, dim2, dim3, channels = input_shape

    xx_ones = np.ones(np.stack([batch_shape, dim2]), dtype='int32')

    xx_range = np.tile(np.expand_dims(np.arange(0, dim3), axis=1), np.stack([batch_shape, 1]))

    xx_channels = np.dot(xx_range, xx_ones)
    xx_channels = np.expand_dims(xx_channels, axis=-1)
    xx_channels = np.transpose(xx_channels, [1, 0, 2])
    xx_channels = np.expand_dims(xx_channels, axis=0)
    xx_channels = np.tile(xx_channels, [dim1, 1, 1, 1])

    yy_ones = np.ones(np.stack([dim3, batch_shape]), dtype='int32')

    yy_range = np.tile(np.expand_dims(np.arange(0, dim2), axis=0), np.stack([batch_shape, 1]))

    yy_channels = np.dot(yy_ones, yy_range)
    yy_channels = np.expand_dims(yy_channels, axis=-1)
    yy_channels = np.transpose(yy_channels, [1, 0, 2])
    yy_channels = np.expand_dims(yy_channels, axis=0)
    yy_channels = np.tile(yy_channels, [dim1, 1, 1, 1])

    zz_range = np.tile(np.expand_dims(np.arange(0, dim1), axis=1), np.stack([batch_shape, 1]))
    zz_range = np.expand_dims(zz_range, axis=-1)

    zz_channels = np.tile(zz_range, [1, dim2, dim3])
    zz_channels = np.expand_dims(zz_channels, axis=-1)

    xx_channels = xx_channels.astype(np.float32)
    xx_channels = xx_channels / (dim2 - 1)
    xx_channels = xx_channels * 2 - 1.

    yy_channels = yy_channels.astype(np.float32)
    yy_channels = yy_channels / (dim3 - 1)
    yy_channels = yy_channels * 2 - 1.

    zz_channels = zz_channels.astype(np.float32)
    zz_channels = zz_channels / (dim1 - 1)
    zz_channels = zz_channels * 2 - 1.

    outputs = np.concatenate([image, zz_channels, xx_channels, yy_channels],
                             axis=-1)

    return outputs



def fit_cube_param(vol_dim, cube_size, ita):
    dim = np.asarray(vol_dim)
    # cube number and overlap along 3 dimensions
    fold = dim / cube_size + ita
    ovlap = np.ceil(np.true_divide((fold * cube_size - dim), (fold - 1)))
    ovlap = ovlap.astype('int')

    fold = np.ceil(np.true_divide((dim + (fold - 1)*ovlap), cube_size))
    fold = fold.astype('int')

    return fold, ovlap

# decompose volume into list of cubes
def decompose_vol2cube(vol_data, batch_size, cube_size, ita, n_chn=1):
    cube_list = []
    # get parameters for decompose
    fold, ovlap = fit_cube_param(vol_data.shape, cube_size, ita)
    dim = np.asarray(vol_data.shape)
    # decompose
    for R in range(0, fold[0]):
        r_s = R*cube_size - R*ovlap[0]
        r_e = r_s + cube_size
        if r_e >= dim[0]:
            r_s = dim[0] - cube_size
            r_e = r_s + cube_size
        for C in range(0, fold[1]):
            c_s = C*cube_size - C*ovlap[1]
            c_e = c_s + cube_size
            if c_e >= dim[1]:
                c_s = dim[1] - cube_size
                c_e = c_s + cube_size
            for H in range(0, fold[2]):
                h_s = H*cube_size - H*ovlap[2]
                h_e = h_s + cube_size
                if h_e >= dim[2]:
                    h_s = dim[2] - cube_size
                    h_e = h_s + cube_size
                # partition multiple channels
                cube_temp = vol_data[r_s:r_e, c_s:c_e, h_s:h_e]
                cube_batch = np.zeros([batch_size, cube_size, cube_size, cube_size, n_chn]).astype('float32')
                cube_batch[0, :, :, :, 0] = copy.deepcopy(cube_temp)
                # save
                cube_list.append(cube_batch)

    return cube_list



def compose_label_cube2vol(cube_list, vol_dim, cube_size, ita, class_n):
    # get parameters for compose
    fold, ovlap = fit_cube_param(vol_dim, cube_size, ita)
    # create label volume for all classes
    label_classes_mat = (np.zeros([vol_dim[0], vol_dim[1], vol_dim[2], class_n])).astype('int32')
    idx_classes_mat = (np.zeros([cube_size, cube_size, cube_size, class_n])).astype('int32')

    p_count = 0
    for R in range(0, fold[0]):
        r_s = R*cube_size - R*ovlap[0]
        r_e = r_s + cube_size
        if r_e >= vol_dim[0]:
            r_s = vol_dim[0] - cube_size
            r_e = r_s + cube_size
        for C in range(0, fold[1]):
            c_s = C*cube_size - C*ovlap[1]
            c_e = c_s + cube_size
            if c_e >= vol_dim[1]:
                c_s = vol_dim[1] - cube_size
                c_e = c_s + cube_size
            for H in range(0, fold[2]):
                h_s = H*cube_size - H*ovlap[2]
                h_e = h_s + cube_size
                if h_e >= vol_dim[2]:
                    h_s = vol_dim[2] - cube_size
                    h_e = h_s + cube_size
                # histogram for voting
                for k in range(class_n):
                    idx_classes_mat[:, :, :, k] = (cube_list[p_count] == k)
                # accumulation
                label_classes_mat[r_s:r_e, c_s:c_e, h_s:h_e, :] = label_classes_mat[r_s:r_e, c_s:c_e, h_s:h_e, :] + idx_classes_mat

                p_count += 1
    # print 'label mat unique:'
    # print np.unique(label_mat)

    compose_vol = np.argmax(label_classes_mat, axis=-1)
    # print np.unique(label_mat)

    return compose_vol



class TensorBoardWrapper(TensorBoard):
    '''Sets the self.validation_data property for use with TensorBoard callback.'''

    def __init__(self, batch_gen, nb_steps, log_dir, histogram_freq, **kwargs):
        super(TensorBoardWrapper, self).__init__(log_dir, histogram_freq, **kwargs)

        self.batch_gen = batch_gen # The generator.
        self.nb_steps = nb_steps     # Number of times to call next() on the generator.
        self.histogram_freq = histogram_freq

    def set_model(self, model):
        return super(TensorBoardWrapper, self).set_model(model)

    def make_image(self, tensor, convert=False):
        from PIL import Image
        height, width, channel = tensor.shape
        if convert:
            tensor = (tensor * 255)
        tensor = tensor.astype(np.uint8)
        image = Image.fromarray(tensor.reshape(tensor.shape[:-1]))
        import io
        output = io.BytesIO()
        image.save(output, format='PNG')
        image_string = output.getvalue()
        output.close()
        return tf.Summary.Image(height=height,
                                width=width,
                                colorspace=channel,
                                encoded_image_string=image_string)


    def on_epoch_end(self, epoch, logs):
        # Fill in the `validation_data` property. Obviously this is specific to how your generator works.
        # Below is an example that yields images and classification tags.
        # After it's filled in, the regular on_epoch_end method has access to the validation_data.
        imgs, tags = None, None
        for s in range(self.nb_steps):
            ib, tb = next(self.batch_gen)
            if imgs is None and tags is None:

                imgs = np.zeros(((self.nb_steps * ib.shape[0],) + ib.shape[1:]), dtype=np.float32)
                tags = np.zeros(((self.nb_steps * tb.shape[0],) + tb.shape[1:]), dtype=np.uint8)

            imgs[s * ib.shape[0]:(s + 1) * ib.shape[0]] = ib
            tags[s * tb.shape[0]:(s + 1) * tb.shape[0]] = tb

        self.validation_data = [imgs, tags, np.ones(imgs.shape[0])]

        if epoch % self.histogram_freq == 0:
            val_data = self.validation_data

            # Load image
            valid_images = val_data[0]  # X_train
            valid_labels = val_data[1]  # Y_train
            valid_labels = np.argmax(valid_labels, axis=-1)
            valid_labels = valid_labels.reshape(valid_labels.shape + (1,))
            pred_images = self.model.predict(valid_images, batch_size=1)
            pred_images = np.argmax(pred_images, axis=-1)
            pred_images = pred_images.reshape(pred_images.shape + (1,))

            summary_str = list()
            half = int(valid_images.shape[0] / 2)
            for i in tqdm(range(len(pred_images))):
                valid_image = self.make_image(valid_images[i, half, :, :, :], convert=True)
                valid_label = self.make_image(valid_labels[i, half, :, :, :], convert=True)
                pred_image = self.make_image(pred_images[i, half, :, :, :], convert=True)

                summary_str.append(tf.Summary.Value(tag='plot/image/x/%d' % i, image=valid_image))
                summary_str.append(tf.Summary.Value(tag='plot/label/x/%d' % i, image=valid_label))
                summary_str.append(tf.Summary.Value(tag='plot/pred/x/%d' % i, image=pred_image))

                valid_image = self.make_image(valid_images[i, :, half, :, :], convert=True)
                valid_label = self.make_image(valid_labels[i, :, half, :, :], convert=True)
                pred_image = self.make_image(pred_images[i, :, half, :, :], convert=True)

                summary_str.append(tf.Summary.Value(tag='plot/image/y/%d' % i, image=valid_image))
                summary_str.append(tf.Summary.Value(tag='plot/label/y/%d' % i, image=valid_label))
                summary_str.append(tf.Summary.Value(tag='plot/pred/y/%d' % i, image=pred_image))

                valid_image = self.make_image(valid_images[i, :, :, half, :], convert=True)
                valid_label = self.make_image(valid_labels[i, :, :, half, :], convert=True)
                pred_image = self.make_image(pred_images[i, :, :, half, :], convert=True)

                summary_str.append(tf.Summary.Value(tag='plot/image/z/%d' % i, image=valid_image))
                summary_str.append(tf.Summary.Value(tag='plot/label/z/%d' % i, image=valid_label))
                summary_str.append(tf.Summary.Value(tag='plot/pred/z/%d' % i, image=pred_image))


            self.writer.add_summary(tf.Summary(value = summary_str), epoch)

        return super(TensorBoardWrapper, self).on_epoch_end(epoch, logs)