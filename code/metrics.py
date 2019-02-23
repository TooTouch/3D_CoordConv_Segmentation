from functools import partial

from keras import backend as K


def average_dice_coefficient(y_true, y_pred, smooth=1.):
    dice = 0
    for i in range(1,8):
        y_truei = y_true[:, :, :, :, i]
        y_predi = y_pred[:, :, :, :, i]
        intersection = K.sum(y_truei * y_predi)
        dice = dice + (2. * intersection + smooth) / (K.sum(y_truei) + K.sum(y_predi) + smooth)
    return dice / 7.


def dice_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)


# def weighted_dice_coefficient_loss(y_true, y_pred, smooth=1.):
#     """
#     Weighted dice coefficient. Default axis assumes a "channels first" data structure
#     :param y_true:
#     :param y_pred:
#     :return:
#     """
#     dice = 0
#     for i in range(8):
#         y_truei = y_true[:, :, :, :, i]
#         y_predi = y_pred[:, :, :, :, i]
#         weight = 1 - K.sum(y_truei)/K.sum(y_true)
#         d = weight * (2 * K.sum(y_truei * y_predi) + smooth) / (K.sum(y_truei) + K.sum(y_predi) + smooth)
#         dice = dice - K.log(d)
#     return dice

def weighted_dice_coefficient_loss(y_true, y_pred, axis=(1, 2, 3), smooth=0.00001):
    """
    Weighted dice coefficient. Default axis assumes a "channels first" data structure
    :param smooth:
    :param y_true:
    :param y_pred:
    :param axis:
    :return:
    """
    return 1 - K.mean(2. * (K.sum(y_true * y_pred, axis=axis) + smooth/2)/(K.sum(y_true,  axis=axis) + K.sum(y_pred, axis=axis) + smooth))


def label_wise_dice_coefficient(y_true, y_pred, label_index):
    return dice_coefficient(y_true[:,:,:,:,label_index], y_pred[:,:,:,:,label_index])


def get_label_dice_coefficient_function(label_index):
    f = partial(label_wise_dice_coefficient, label_index=label_index)
    f.__setattr__('__name__', 'DSC_{0}'.format(label_index))
    return f

def softmax_weighted_loss(y_true, y_pred):
    """
    Loss = weighted * -target*log(softmax(logits))
    :param logits: probability score
    :param labels: ground_truth
    :return: softmax-weifhted loss
    """

    loss = 0
    for i in range(8):
        y_truei = y_true[:, :, :, :, i]
        y_predi = y_pred[:, :, :, :, i]
        weighted = 1 - (K.sum(y_truei) / K.sum(y_true))
        loss = loss - weighted * K.mean(y_truei * K.log(K.clip(y_predi, 0.005, 1)))
    return loss

def combine_loss(y_true, y_pred):
    return softmax_weighted_loss(y_pred, y_true) + weighted_dice_coefficient_loss(y_true, y_pred)

