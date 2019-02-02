import numpy as np
import copy
import nibabel as nib
from skimage.transform import resize
from keras.utils import to_categorical

def load_data_pairs(pair_list, resize_r, output_chn, rename_map):
    """load all volume pairs"""
    img_clec = []
    label_clec = []

    for k in range(0, len(pair_list), 2):
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

        img_clec.append(img_data)
        label_clec.append(lab_r_data)

    return img_clec, label_clec



def get_batch_patches(img, label, patch_dim, output_chn, batch_size, chn=1, flip_flag=True, rot_flag=True):
    """generate a batch of paired patches for training"""
    batch_img = np.zeros([batch_size, patch_dim, patch_dim, patch_dim, chn]).astype('float32')
    batch_label = np.zeros([batch_size, patch_dim, patch_dim, patch_dim, output_chn]).astype('int32')

    for k in range(batch_size):
        rand_img = img
        rand_label = label
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

        label_temp = copy.deepcopy(rand_label[pos[0]:pos[0] + patch_dim, pos[1]:pos[1] + patch_dim, pos[2]:pos[2] + patch_dim, :])

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