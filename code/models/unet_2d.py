import keras.layers as KL
import keras.models as KM
import keras.optimizers as KO

from metrics import dice_coefficient_loss, get_label_dice_coefficient_function, dice_coefficient, weighted_dice_coefficient_loss

class Unet2d:
    def __init__(self, input_shape, pool_size=(2, 2), n_labels=1, initial_learning_rate=0.00001, pretrained_weights=None):
        self.input_shape = input_shape
        self.pool_size = pool_size
        self.n_labels = n_labels
        self.initial_learning_rate = initial_learning_rate
        self.metrics = dice_coefficient

        if self.n_labels == 1:
            self.activation_name = 'sigmoid'
            self.include_label_wise_dice_coefficients = False
        else:
            self.activation_name = 'softmax'
            self.include_label_wise_dice_coefficients = True
        self.pretrained_weights = pretrained_weights


    def build(self):
        inputs = KL.Input(self.input_shape)
        conv1 = KL.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = KL.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = KL.MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = KL.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = KL.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = KL.MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = KL.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = KL.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = KL.MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = KL.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = KL.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = KL.Dropout(0.5)(conv4)
        pool4 = KL.MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = KL.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = KL.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = KL.Dropout(0.5)(conv5)

        up6 = KL.Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(KL.UpSampling2D(size=(2, 2))(drop5))
        merge6 = KL.concatenate([drop4, up6], axis=1)
        conv6 = KL.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = KL.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = KL.Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(KL.UpSampling2D(size=(2, 2))(conv6))
        merge7 = KL.concatenate([conv3, up7], axis=1)
        conv7 = KL.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = KL.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        up8 = KL.Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(KL.UpSampling2D(size=(2, 2))(conv7))
        merge8 = KL.concatenate([conv2, up8], axis=1)
        conv8 = KL.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = KL.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = KL.Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(KL.UpSampling2D(size=(2, 2))(conv8))
        merge9 = KL.concatenate([conv1, up9], axis=1)
        conv9 = KL.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = KL.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = KL.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv10 = KL.Conv2D(self.n_labels, 1, activation=self.activation_name)(conv9)

        model = KM.Model(input=inputs, output=conv10)

        if not isinstance(self.metrics, list):
            self.metrics = [self.metrics]

        if self.include_label_wise_dice_coefficients and self.n_labels > 1:
            label_wise_dice_metrics = [get_label_dice_coefficient_function(index) for index in range(self.n_labels)]
            if self.metrics:
                self.metrics = self.metrics + label_wise_dice_metrics
            else:
                self.metrics = label_wise_dice_metrics


        model.compile(optimizer=KO.Adam(lr=self.initial_learning_rate), loss=weighted_dice_coefficient_loss, metrics=self.metrics)

        # model.summary()

        if (self.pretrained_weights):
            model.load_weights(self.pretrained_weights)

        return model