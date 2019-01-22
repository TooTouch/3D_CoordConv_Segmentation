import os
import numpy as np

from keras import backend as K
from keras.engine import Input, Model
from keras import layers as KL
from keras.optimizers import Adam
from keras.models import load_model

from metrics import *

K.set_image_data_format("channels_first")

try:
    from keras.engine import merge
except ImportError:
    from keras.layers.merge import concatenate


class Unet3d:
    def __init__(self, input_shape, pool_size=(2, 2, 2), n_labels=1, initial_learning_rate=0.00001, deconvolution=False,
                      depth=4, n_base_filters=32, batch_normalization=False, pretrained_weights=None, pretrained_model='None'):
        '''

        :param input_shape: Shape of the input data (x_size, y_size, z_size, n_chanels). The x, y, and z sizes must be
                            divisible by the pool size to the power of the depth of the UNet, that is pool_size^depth.
        :param pool_size: Pool size for the max pooling operations.
        :param n_labels: Number of binary labels that the model is learning.
        :param initial_learning_rate: Initial learning rate for the model. This will be decayed during training.
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
        self.initial_learning_rate = initial_learning_rate
        self.deconvolution = deconvolution,
        self.depth = depth
        self.n_base_filters = n_base_filters
        self.metrics = dice_coefficient
        self.batch_normalization = batch_normalization
        self.model_dir = os.path.abspath(os.path.join(os.getcwd(),'../model/'+pretrained_model+'.h5'))
        if self.n_labels == 1:
            self.activation_name = 'sigmoid'
            self.include_label_wise_dice_coefficients = False
        else:
            self.activation_name = 'softmax'
            self.include_label_wise_dice_coefficients = True

        self.pretrained_weights = pretrained_weights

    def build(self):
        inputs = Input(self.input_shape)
        current_layer = inputs
        levels = list()

        # add levels with max pooling
        for layer_depth in range(self.depth):
            layer1 = self.create_convolution_block(input_layer=current_layer, n_filters=self.n_base_filters*(2**layer_depth))
            layer2 = self.create_convolution_block(input_layer=layer1, n_filters=self.n_base_filters*(2**layer_depth))
            if layer_depth < self.depth - 1:
                current_layer = KL.MaxPooling3D(pool_size=self.pool_size, data_format = 'channels_first')(layer2)
                levels.append([layer1, layer2, current_layer])
            else:
                current_layer = layer2
                levels.append([layer1, layer2])

        # add levels with up-convolution or up-sampling
        for layer_depth in range(self.depth-2, -1, -1):
            up_convolution = self.get_up_convolution(pool_size=self.pool_size,
                                                     n_filters=self.n_base_filters*(2**layer_depth))(current_layer)
            concat = concatenate([up_convolution, levels[layer_depth][1]], axis=1)
            current_layer = self.create_convolution_block(n_filters=self.n_base_filters*(2**layer_depth),
                                                     input_layer=concat)
            current_layer = self.create_convolution_block(n_filters=self.n_base_filters*(2**layer_depth),
                                                     input_layer=current_layer)

            current_layer = KL.Conv3D(32, (1, 1, 1))(current_layer)
            current_layer = KL.Activation('relu')(current_layer)
        final_convolution = KL.Conv3D(self.n_labels, (1, 1, 1))(current_layer)
        act = KL.Activation(self.activation_name)(final_convolution)
        model = Model(inputs=inputs, outputs=act)

        if (self.pretrained_weights):
            model.load_weights(self.pretrained_weights)

        if not isinstance(self.metrics, list):
            self.metrics = [self.metrics]

        if self.include_label_wise_dice_coefficients and self.n_labels > 1:
            label_wise_dice_metrics = [get_label_dice_coefficient_function(index) for index in range(self.n_labels)]
            if self.metrics:
                self.metrics = self.metrics + label_wise_dice_metrics
            else:
                self.metrics = label_wise_dice_metrics


        model.compile(optimizer=Adam(lr=self.initial_learning_rate), loss=softmax_weighted_loss, metrics=self.metrics)

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
        layer = KL.Conv3D(n_filters, kernel, padding=padding, strides=strides)(input_layer)
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


    def compute_level_output_shape(self, n_filters, depth, pool_size, image_shape):
        """
        Each level has a particular output shape based on the number of filters used in that level and the depth or number
        of max pooling operations that have been done on the data at that point.
        :param image_shape: shape of the 3d image.
        :param pool_size: the pool_size parameter used in the max pooling operation.
        :param n_filters: Number of filters used by the last node in a given level.
        :param depth: The number of levels down in the U-shaped model a given node is.
        :return: 5D vector of the shape of the output node
        """
        output_image_shape = np.asarray(np.divide(image_shape, np.power(pool_size, depth)), dtype=np.int32).tolist()
        return tuple([None, n_filters] + output_image_shape)


    def get_up_convolution(self, n_filters, pool_size, kernel_size=(2, 2, 2), strides=(2, 2, 2)):
        if self.deconvolution:
            return KL.Deconvolution3D(filters=n_filters, kernel_size=kernel_size,
                                   strides=strides)
        else:
            return KL.UpSampling3D(size=pool_size)

    def finetune_model(self):
        model = load_model(self.model_dir,
                                  custom_objects={'weighted_dice_coefficient_loss': weighted_dice_coefficient_loss,
                                                  'dice_coefficient': dice_coefficient})

        x = KL.Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same', activation='relu', name='fine_tune_conv3d_0')(model.layers[-3].output)
        x = KL.Conv3D(filters=16, kernel_size=(3, 3, 3), padding='same', activation='relu', name='fine_tune_conv3d_1')(x)
        output = KL.Conv3D(filters=8, kernel_size=(3, 3, 3), padding='same', activation='softmax',name='fine_tune_conv3d_2')(x)
        model = Model(inputs=model.input, outputs=output)

        if not isinstance(self.metrics, list):
            self.metrics = [self.metrics]

        if self.include_label_wise_dice_coefficients and self.n_labels > 1:
            label_wise_dice_metrics = [get_label_dice_coefficient_function(index) for index in range(self.n_labels)]
            if self.metrics:
                self.metrics = self.metrics + label_wise_dice_metrics
            else:
                self.metrics = label_wise_dice_metrics

        model.compile(optimizer=Adam(lr=self.initial_learning_rate), loss=weighted_dice_coefficient_loss,
                      metrics=self.metrics)

        return model