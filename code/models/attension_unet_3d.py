from keras import layers as KL
from keras import models as KM
import keras.backend as K

def Unet_Attention(self):
    def expend_as(tensor, rep):
        my_repeat = KL.Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=-1), arguments={'repnum': rep})(tensor)
        return my_repeat

    def UnetGatingSignal(input):
        shape = K.int_shape(input)
        # print('gating signal >>>>> ', shape)
        x = KL.Conv2D(shape[3] * 2, (1, 1), strides=(1, 1), padding="same")(input)
        x = KL.BatchNormalization()(x)
        x = KL.Activation('relu')(x)
        return x

    def AttentionGate(x, g, inter_shape):
        '''
        x : input
        g : gate
        '''
        shape_x = K.int_shape(x)  # 32
        shape_g = K.int_shape(g)  # 16
        # print('shape_x >>>>> ', shape_x)
        # print('shape_g >>>>> ', shape_g)
        #
        theta_x = KL.Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)  # 16
        shape_theta_x = K.int_shape(theta_x)
        # print('shape_theta_x >>>>> ', shape_theta_x)

        phi_g = KL.Conv2D(inter_shape, (1, 1), padding='same')(g)
        # print('phi_g >>>>> ', phi_g)
        upsample_g = KL.Conv2DTranspose(inter_shape, (3, 3),
                                        strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
                                        padding='same')(phi_g)  # 16
        # print('upsample_g >>>>> ', upsample_g)

        concat_xg = KL.merge.add([upsample_g, theta_x])
        act_xg = KL.Activation('relu')(concat_xg)
        psi = KL.Conv2D(1, (1, 1), padding='same')(act_xg)
        sigmoid_xg = KL.Activation('sigmoid')(psi)
        shape_sigmoid = K.int_shape(sigmoid_xg)
        upsample_psi = KL.UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(
            sigmoid_xg)  # 32

        # my_repeat=Lambda(lambda xinput:K.repeat_elements(xinput[0],shape_x[1],axis=1))
        # upsample_psi=my_repeat([upsample_psi])
        upsample_psi = expend_as(upsample_psi, shape_x[3])
        y = KL.merge.multiply([upsample_psi, x])

        # print(K.is_keras_tensor(upsample_psi))

        result = KL.Conv2D(shape_x[3], (1, 1), padding='same')(y)
        result_bn = KL.BatchNormalization()(result)
        return result_bn

    def UnetConv2D(input, outdim, is_batchnorm=False):
        x = KL.Conv2D(outdim, (3, 3), strides=(1, 1), padding="same")(input)
        if is_batchnorm:
            x = KL.BatchNormalization()(x)
        x = KL.Activation('relu')(x)

        x = KL.Conv2D(outdim, (3, 3), strides=(1, 1), padding="same")(x)
        if is_batchnorm:
            x = KL.BatchNormalization()(x)
        x = KL.Activation('relu')(x)
        return x

    self.inputs = KL.Input(shape=self.input_shape)
    s = KL.Lambda(lambda x: x / 255)(self.inputs)

    # Block 1
    c1 = UnetConv2D(s, 64, True)
    p1 = KL.MaxPooling2D(pool_size=(2, 2), strides=2)(c1)

    # Block 2
    c2 = UnetConv2D(p1, 128, True)
    p2 = KL.MaxPooling2D(pool_size=(2, 2), strides=2)(c2)

    # Block 3
    c3 = UnetConv2D(p2, 256, True)
    p3 = KL.MaxPooling2D(pool_size=(2, 2), strides=2)(c3)

    # Block 4
    c4 = UnetConv2D(p3, 512, True)
    p4 = KL.MaxPooling2D(pool_size=(2, 2), strides=2)(c4)

    # Block 5
    c5 = UnetConv2D(p4, 1024, True)

    # Block 6
    g6 = UnetGatingSignal(c5)
    att6 = AttentionGate(c4, g6, 512)
    up6 = KL.Conv2DTranspose(512, (3, 3), activation='relu', strides=(2, 2), padding='same')(c5)
    up6 = KL.merge.concatenate([up6, att6])
    c6 = UnetConv2D(up6, 512, True)

    # Block 7
    g7 = UnetGatingSignal(c6)
    att7 = AttentionGate(c3, g7, 256)
    up7 = KL.Conv2DTranspose(256, (3, 3), activation='relu', strides=(2, 2), padding='same')(c6)
    up7 = KL.merge.concatenate([up7, att7])
    c7 = UnetConv2D(up7, 256, True)

    # Block 8
    g8 = UnetGatingSignal(c7)
    att8 = AttentionGate(c2, g8, 128)
    up8 = KL.Conv2DTranspose(128, (3, 3), activation='relu', strides=(2, 2), padding='same')(c7)
    up8 = KL.merge.concatenate([up8, att8])
    c8 = UnetConv2D(up8, 128, True)

    # Block 9
    g9 = UnetGatingSignal(c8)
    att9 = AttentionGate(c1, g9, 64)
    up9 = KL.Conv2DTranspose(64, (3, 3), activation='relu', strides=(2, 2), padding='same')(c8)
    up9 = KL.merge.concatenate([up9, att9])
    c9 = UnetConv2D(up9, 64, True)

    c10 = KL.Conv2D(self.num_classes, 1)(c9)
    r10 = KL.Reshape((-1, self.num_classes))(c10)
    self.logits = KL.Activation('softmax')(r10)

    model = KM.Model(self.inputs, self.logits, name='Unet_Attention')

    return model