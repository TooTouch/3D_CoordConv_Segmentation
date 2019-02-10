import os
import numpy as np

from keras import backend as K
from keras.engine import Input, Model
from keras import layers as KL
from keras.optimizers import Adam,  RMSprop
from keras.models import load_model

from metrics import *

#K.set_image_data_format("channels_first")
K.set_image_data_format("channels_last")

try:
    from keras.engine import merge
except ImportError:
    from keras.layers.merge import concatenate


class Unet3d:
    def __init__(self, input_shape, pool_size=(2, 2, 2), n_labels=1, lrate=0.00001, deconvolution=False,
                      depth=4, n_base_filters=32, batch_normalization=False):
        '''

        :param input_shape: Shape of the input data (x_size, y_size, z_size, n_chanels). The x, y, and z sizes must be
                            divisible by the pool size to the power of the depth of the UNet, that is pool_size^depth.
        :param pool_size: Pool size for the max pooling operations.
        :param n_labels: Number of binary labels that the model is learning.
        :param lrate: Initial learning rate for the model. This will be decayed during training.
        :param deconvolution: If set to True, will use transpose convolution(deconvolution) instead of up-sampling. This
                                increases the amount memory required during training.
        :param depth: indicates the depth of the U-shape for the model. The greater the depth, the more max pooling
                        layers will be added to the model. Lowering the depth may reduce the amount of memory required for training.
        :param n_base_filters: The number of filters that the first layer in the convolution network will have. Following
                layers will contain a multiple of this number. Lowering this number will likely reduce the amount of memory required
                to train the model.
        :param batch_normalization: 
        :param pretrained_weights:
        :param pretrained_model:
        '''

        self.input_shape = input_shape
        print('Input shape: ',self.input_shape)
        self.pool_size = pool_size
        self.n_labels = n_labels
        self.lrate = lrate
        self.deconvolution = deconvolution,
        self.depth = depth
        self.n_base_filters = n_base_filters
        self.metrics = dice_coefficient
        self.loss = dice_coefficient_loss
        self.batch_normalization = batch_normalization
        if self.n_labels == 1:
            self.activation_name = 'sigmoid'
            self.include_label_wise_dice_coefficients = False
        else:
            self.activation_name = 'softmax'
            self.include_label_wise_dice_coefficients = True


    def build(self):
        inputs = Input(self.input_shape)
        # Encoder
        en_conv1_1 = self.create_convolution_block(input_layer=inputs, n_filters=self.n_base_filters*2)
        en_conv1_2 = self.create_convolution_block(input_layer=en_conv1_1, n_filters=self.n_base_filters*2)
        pool1 = KL.MaxPooling3D(pool_size=self.pool_size, data_format='channels_last')(en_conv1_2)

        en_conv2_1 = self.create_convolution_block(input_layer=pool1, n_filters=self.n_base_filters*(2**2))
        en_conv2_2 = self.create_convolution_block(input_layer=en_conv2_1, n_filters=self.n_base_filters*(2**2))
        pool2 = KL.MaxPooling3D(pool_size=self.pool_size, data_format='channels_last')(en_conv2_2)

        en_conv3_1 = self.create_convolution_block(input_layer=pool2, n_filters=self.n_base_filters * (2 ** 3))
        en_conv3_2 = self.create_convolution_block(input_layer=en_conv3_1, n_filters=self.n_base_filters * (2 ** 3))
        pool3 = KL.MaxPooling3D(pool_size=self.pool_size, data_format='channels_last')(en_conv3_2)

        en_conv4_1 = self.create_convolution_block(input_layer=pool3, n_filters=self.n_base_filters * (2 ** 4))
        en_conv4_2 = self.create_convolution_block(input_layer=en_conv4_1, n_filters=self.n_base_filters * (2 ** 4))

        # Decoder
        deconv1 = self.get_up_convolution(pool_size=self.pool_size, n_filters=self.n_base_filters * (2 ** 3))(en_conv4_2)
        concat1 = concatenate([deconv1, en_conv3_2], axis=-1)
        ########### 이 부분 변경
        de_conv1_1 =  self.create_convolution_block(input_layer=concat1, n_filters=self.n_base_filters * (2 ** 3))
        de_conv1_2 =  self.create_convolution_block(input_layer=de_conv1_1, n_filters=self.n_base_filters * (2 ** 3))

        deconv2 = self.get_up_convolution(pool_size=self.pool_size, n_filters=self.n_base_filters * (2 ** 2))(de_conv1_2)
        concat2 = concatenate([deconv2, en_conv2_2], axis=-1)
        de_conv2_1 = self.create_convolution_block(input_layer=concat2, n_filters=self.n_base_filters * (2 ** 2))
        de_conv2_2 = self.create_convolution_block(input_layer=de_conv2_1, n_filters=self.n_base_filters * (2 ** 2))

        deconv3 = self.get_up_convolution(pool_size=self.pool_size, n_filters=self.n_base_filters * 2)(de_conv2_2)
        concat3 = concatenate([deconv3, en_conv1_2], axis=-1)
        de_conv3_1 = self.create_convolution_block(input_layer=concat3, n_filters=self.n_base_filters * 2)
        de_conv3_2 = self.create_convolution_block(input_layer=de_conv3_1, n_filters=self.n_base_filters * 2)

        # output
        conv = self.create_convolution_block(input_layer=de_conv3_2, n_filters=self.n_base_filters)
        output = KL.Conv3D(self.n_labels, (1, 1, 1), activation=self.activation_name, data_format='channels_last', name='output_layer')(conv)

        model = Model(inputs=inputs, outputs=output)

        if not isinstance(self.metrics, list):
            self.metrics = [self.metrics]
        if not isinstance(self.loss, list):
            self.loss = [self.loss]

        if self.include_label_wise_dice_coefficients and self.n_labels > 1:
            label_wise_dice_metrics = [get_label_dice_coefficient_function(index) for index in range(self.n_labels)]
            if self.metrics:
                self.metrics = self.metrics + label_wise_dice_metrics
            else:
                self.metrics = label_wise_dice_metrics
            if self.loss:
                self.loss = label_wise_dice_coefficient_loss

        model.compile(optimizer=RMSprop(lr=self.lrate), loss=softmax_weighted_loss, metrics=self.metrics)

        return model



    def create_convolution_block(self, input_layer, n_filters, kernel=(3, 3, 3), activation=None,
                                 padding='same', strides=(1, 1, 1), instance_normalization=False):
        """
        :param strides:
        :param input_layer:
        :param n_filters:
        :param batch_normalization:
        :param kernel:
        :param activation: Keras activation layer to use. (default is 'relu')
        :param padding:
        :return:
        """
        layer = KL.Conv3D(n_filters, kernel, padding=padding, strides=strides, data_format='channels_last')(input_layer)
        if self.batch_normalization:
            layer = KL.BatchNormalization(axis=1)(layer)
        elif instance_normalization:
            try:
                from keras_contrib.layers.normalization import InstanceNormalization
            except ImportError:
                raise ImportError("Install keras_contrib in order to use instance normalization."
                                  "\nTry: pip install git+https://www.github.com/farizrahman4u/keras-contrib.git")
            layer = InstanceNormalization(axis=1)(layer)
        if activation is None:
            return KL.Activation('relu')(layer)
        else:
            return activation()(layer)


    def get_up_convolution(self, n_filters, pool_size, kernel_size=(2, 2, 2), strides=(2, 2, 2)):
        if self.deconvolution:
            return KL.Deconvolution3D(filters=n_filters, kernel_size=kernel_size,
                                   strides=strides, data_format='channels_last')
        else:
            return KL.UpSampling3D(size=pool_size)
