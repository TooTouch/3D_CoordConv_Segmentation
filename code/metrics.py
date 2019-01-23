from functools import partial

from keras import backend as K


def dice_coefficient(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))


def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)


def weighted_dice_coefficient(y_true, y_pred, axis=(-3, -2, -1), smooth=0.00001):
    """
    Weighted dice coefficient. Default axis assumes a "channels first" data structure
    :param smooth:
    :param y_true:
    :param y_pred:
    :param axis:
    :return:
    """
    return K.mean(2. * (K.sum(y_true * y_pred, axis=axis) + smooth/2)/(K.sum(y_true, axis=axis) + K.sum(y_pred, axis=axis) + smooth))


def weighted_dice_coefficient_loss(y_true, y_pred):
    return -weighted_dice_coefficient(y_true, y_pred)

def label_wise_dice_coefficient_loss(y_true, y_pred):
    return K.sum([-dice_coefficient(y_true[:,:,:,:,i], y_pred[:,:,:,:,i]) for label_index in range(8)])


def get_label_dice_coefficient_loss_function(label_index):
    f = partial(label_wise_dice_coefficient, label_index=label_index)
    f.__setattr__('__name__', 'DSC_{0}_loss'.format(label_index))
    return f

def label_wise_dice_coefficient(y_true, y_pred, label_index):
    return dice_coefficient(y_true[:,:,:,:,label_index], y_pred[:,:,:,:,label_index])


def get_label_dice_coefficient_function(label_index):
    f = partial(label_wise_dice_coefficient, label_index=label_index)
    f.__setattr__('__name__', 'DSC_{0}'.format(label_index))
    return f

def categorical_crossentropy(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred)

def softmax_weighted_loss(labels, logits):
    """
    Loss = weighted * -target*log(softmax(logits))
    :param logits: probability score
    :param labels: ground_truth
    :return: softmax-weifhted loss
    """

    gt = labels
    softmaxpred = logits
    loss = 0
    # labels = K.print_tensor(labels, message='labels.shape: ')
    # logits = K.print_tensor(logits, message='logits.shape: ')
    for i in range(8):
        gti = gt[:, :, :, :, i]
        predi = softmaxpred[:, :, :, :, i]
        weighted = 1 - (K.sum(gti) / K.sum(gt))
        # print("class %d"%(i) )
        # print(weighted)
        loss = loss + -K.mean(weighted * gti * K.log(K.clip(predi, 0.005, 1)))
    return loss

def combine_loss(y_true, y_pred):
    return softmax_weighted_loss(y_pred, y_true) + weighted_dice_coefficient(y_true, y_pred)


dice_coef = dice_coefficient
dice_coef_loss = dice_coefficient_loss
