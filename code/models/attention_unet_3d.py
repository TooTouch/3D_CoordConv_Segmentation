import os
import numpy as np

from keras import backend as K
from keras.engine import Input, Model
from keras import layers as KL
from keras.optimizers import Adam
from keras.models import load_model
from keras.utils import multi_gpu_model

from Normalization import *
from CoordNet import *
from metrics import *

try:
    from keras.engine import merge
except ImportError:
    from keras.layers.merge import concatenate


class AttnUnet3d:
    def __init__(self, input_size, loss, pool_size=(2, 2, 2), n_labels=1, lrate=0.00001, deconvolution=False,
                      n_base_filters=32, normalization=None, coordnet=False):
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

        self.input_size = input_size
        self.pool_size = pool_size
        self.n_labels = n_labels
        self.lrate = lrate
        self.deconvolution = deconvolution
        self.n_base_filters = n_base_filters # 32 
        self.metrics = average_dice_coefficient
        self.loss = loss
        self.normalization = normalization
        self.coordnet = coordnet
        if self.n_labels == 1:
            self.activation_name = 'sigmoid'
            self.include_label_wise_dice_coefficients = False
        else:
            self.activation_name = 'softmax'
            self.include_label_wise_dice_coefficients = True


    def expand_as(self, tensor, repeat):
        my_repeat = KL.Lambda(lambda x, repnum: K.repeat_elements(x,repnum,axis=4),arguments={'repnum':repeat})(tensor)
        return my_repeat 

    def attn_gating_block(self, input_layer, gating_signal, inter_shape):
        shape_input = K.int_shape(input_layer) # input_layer : None, 12,12,12,512
        shape_gating =  K.int_shape(gating_signal) #gating signal None,6,6,6,12 
        print("shape_input:",shape_input,"shape_gating:",shape_gating)

        theta_input = KL.Conv3D(inter_shape, kernel_size=(2,2,2), strides=(2,2,2), padding='same')(input_layer)
        shape_theta_input = K.int_shape(theta_input) 

        phi_gating = KL.Conv3D(filters=inter_shape, kernel_size=(1,1,1), padding='same')(gating_signal)
        upsample_gating = KL.Conv3DTranspose(inter_shape, kernel_size=(3,3,3), strides=(shape_theta_input[1]//shape_gating[1],shape_theta_input[2]//shape_gating[2],shape_theta_input[3]//shape_gating[3]), padding='same')(phi_gating)

        concat_lg=KL.add([upsample_gating, theta_input])
        act_lg=KL.Activation('relu')(concat_lg)
        psi = KL.Conv3D(filters=1, kernel_size=(1,1,1),padding='same')(act_lg)
        sigmoid_lg=KL.Activation('sigmoid')(psi)
        shape_sigmoid = K.int_shape(sigmoid_lg)
        upsample_psi = KL.UpSampling3D(size=(shape_input[1]//shape_sigmoid[1], shape_input[2]//shape_sigmoid[2], shape_input[3]//shape_sigmoid[3]))(sigmoid_lg)
        # upsample_psi 
        upsample_psi = self.expand_as(upsample_psi, shape_input[4]) 

        y = KL.multiply([upsample_psi, input_layer])

        result = KL.Conv3D(filters=shape_input[3],kernel_size=(1,1,1),padding="same")(y)
        result_bn = KL.BatchNormalization()(result)

        return result_bn

    def unet_gating_signal(self, input_layer, is_batchnorm=False):
        shape = K.int_shape(input_layer)
        #layer = KL.Conv3D(shape[3]*2,(1,1,1),strides=(1,1,1), padding="same")(input_layer)
        print(shape, input_layer)
        layer = KL.Conv3D(shape[4] * 2, (1, 1, 1), strides=(1, 1, 1), padding="same")(input_layer)
        if is_batchnorm:
            layer = KL.BatchNormalization()(layer)
        layer = KL.Activation('relu')(layer)
        print("unet_gating_signal: layer:",layer)

        return layer 



    def build(self, input_chn, multi_gpu=1):
        input_shape = ((self.input_size,) * 3 + (input_chn,))
        inputs = Input(input_shape)
        # Encoder
        en_conv1_1 = self.conv_block(input_layer=inputs, n_filters=self.n_base_filters*2, coordnet=self.coordnet) #64 
        en_conv1_2 = self.conv_block(input_layer=en_conv1_1, n_filters=self.n_base_filters*2)
        pool1 = KL.MaxPooling3D(pool_size=self.pool_size, data_format='channels_last')(en_conv1_2)

        en_conv2_1 = self.conv_block(input_layer=pool1, n_filters=self.n_base_filters*(2**2)) #128 
        en_conv2_2 = self.conv_block(input_layer=en_conv2_1, n_filters=self.n_base_filters*(2**2))
        pool2 = KL.MaxPooling3D(pool_size=self.pool_size, data_format='channels_last')(en_conv2_2)

        en_conv3_1 = self.conv_block(input_layer=pool2, n_filters=self.n_base_filters * (2 ** 3)) #256 
        en_conv3_2 = self.conv_block(input_layer=en_conv3_1, n_filters=self.n_base_filters * (2 ** 3))
        pool3 = KL.MaxPooling3D(pool_size=self.pool_size, data_format='channels_last')(en_conv3_2)

        en_conv4_1 = self.conv_block(input_layer=pool3, n_filters=self.n_base_filters * (2 ** 4)) #512 
        en_conv4_2 = self.conv_block(input_layer=en_conv4_1, n_filters=self.n_base_filters * (2 ** 4))
        # add for attn
        pool4 = KL.MaxPooling3D(pool_size=self.pool_size, data_format='channels_last')(en_conv4_2) 

        center = self.conv_block(input_layer=pool4, n_filters=self.n_base_filters*(2**5))#1024  

# self, input_layer, kernel_size, is_batchnorm=False
        gating = self.unet_gating_signal(center)
        attn_1_1 = self.attn_gating_block(input_layer=en_conv4_1, gating_signal=gating, inter_shape=self.n_base_filters*(2**5))
        #  def attn_gating_block(self, input_layer, gating_signal, inter_shape): 
        attn_1_2 = self.attn_gating_block(input_layer=en_conv4_2, gating_signal=gating, inter_shape=self.n_base_filters*(2**5))
        up1 = concatenate([KL.Conv3DTranspose(self.n_base_filters*(2**4),(3,3,3),strides=(2,2,2),padding='same',activation="relu")(center), attn_1_2],axis=-1) 

        gating = self.unet_gating_signal(up1)
        attn_2_1 = self.attn_gating_block(input_layer=en_conv3_1, gating_signal=gating, inter_shape=self.n_base_filters*(2**4))
        attn_2_2 = self.attn_gating_block(input_layer=en_conv3_2, gating_signal=gating, inter_shape=self.n_base_filters*(2**4))
        up2 = concatenate([KL.Conv3DTranspose(self.n_base_filters*(2**4),(3,3,3),strides=(2,2,2),padding='same',activation="relu")(up1), attn_2_2],axis=-1) 

        gating = self.unet_gating_signal(up2)
        attn_3_1 = self.attn_gating_block(input_layer=en_conv2_1, gating_signal=gating, inter_shape=self.n_base_filters*(2**3))
        attn_3_2 = self.attn_gating_block(input_layer=en_conv2_2, gating_signal=gating, inter_shape=self.n_base_filters*(2**3))
        up3 = concatenate([KL.Conv3DTranspose(self.n_base_filters*(2**3),(3,3,3),strides=(2,2,2),padding='same',activation="relu")(up2), attn_3_2],axis=-1)

        up4 = concatenate([KL.Conv3DTranspose(self.n_base_filters*(2**3),(3,3,3),strides=(2,2,2),padding='same',activation="relu")(up3),en_conv1_2],axis=-1)

        #output 
        conv = self.conv_block(input_layer=up4, n_filters=self.n_base_filters)
        output = KL.Conv3D(self.n_labels, (1, 1, 1), activation=self.activation_name, data_format='channels_last')(conv)

        model = Model(inputs=inputs, outputs=output)

        # Metrics and Loss function
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

        model.compile(optimizer=Adam(lr=self.lrate), loss=self.loss, metrics=self.metrics)

         if len(multi_gpu)>1:
            model = multi_gpu_model(model, gpus=len(multi_gpu))
            model.compile(optimizer=Adam(lr=self.lrate), loss=self.loss, metrics=self.metrics)
        

        return model



    def conv_block(self, input_layer, n_filters, kernel=(3, 3, 3), activation=None, padding='same', strides=(1, 1, 1), coordnet=False):
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
        if coordnet:
            input_layer = CoordinateChannel3D()(input_layer)

        layer = KL.Conv3D(n_filters, kernel, padding=padding, strides=strides, data_format='channels_last')(input_layer)
        if self.normalization=='BatchNormalization':
            layer = KL.BatchNormalization(axis=-1)(layer)
        elif self.normalization=='GroupNormalization':
            layer = GroupNormalization(groups=8, axis=-1)(layer)
        elif self.normalization=='InstanceNormalization':
            try:
                from keras_contrib.layers.normalization import InstanceNormalization
            except ImportError:
                raise ImportError("Install keras_contrib in order to use instance normalization."
                                  "\nTry: pip install git+https://www.github.com/farizrahman4u/keras-contrib.git")
            layer = InstanceNormalization(axis=-1)(layer)
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


  
