import torch.nn  as nn
import torch
from torch.autograd import Variable

## unet3D Pytorch Version

def conv3d_block(input_channel, output_channel):
    model = nn.Sequential(
        nn.Conv3d(input_channel, output_channel, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True)
    )
    return model


def trans_conv3d_block(input_channel, output_channel):
    model = nn.Sequential(
        nn.ConvTranspose3d(input_channel, output_channel, kernel_size=3,
                           stride=2, padding=1, output_padding=1),
        nn.ReLU(inplace=True)
    )
    return model



class unet3d(nn.Module):
    def __init__(self, n_base_filters=32, n_labels=8):
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
        super(unet3d, self).__init__()
        #self.input_size = input_size
        #self.pool_size = pool_size
        self.n_labels = n_labels
        #self.lrate = lrate
        self.n_base_filters = n_base_filters

        #self.deconvolution = deconvolution
        #self.depth = depth
        #         self.metrics = dice_coefficient
        #         self.loss = dice_coefficient_loss
        #         self.batch_normalization = batch_normalization

        # if self.n_labels == 1:
        #     self.activation_name = 'sigmoid'
        #     self.include_label_wise_dice_coefficients = False
        # else:
        #     self.activation_name = 'softmax'
        #     self.include_label_wise_dice_coefficients = True

        ## Initializing Layers

        ## Encoder
        self.en_conv1_1 = conv3d_block(1, self.n_base_filters * 2)
        self.en_conv1_2 = conv3d_block(self.n_base_filters * 2, self.n_base_filters * 2)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        self.en_conv2_1 = conv3d_block(self.n_base_filters * 2, self.n_base_filters * (2 ** 2))
        self.en_conv2_2 = conv3d_block(self.n_base_filters * (2 ** 2), self.n_base_filters * (2 ** 2))
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        self.en_conv3_1 = conv3d_block(self.n_base_filters * (2 ** 2), self.n_base_filters * (2 ** 3))
        self.en_conv3_2 = conv3d_block(self.n_base_filters * (2 ** 3), self.n_base_filters * (2 ** 3))
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        self.en_conv4_1 = conv3d_block(self.n_base_filters * (2 ** 3), self.n_base_filters * (2 ** 4))
        self.en_conv4_2 = conv3d_block(self.n_base_filters * (2 ** 4), self.n_base_filters * (2 ** 4))

        ## Decoder
        self.deconv1 = trans_conv3d_block(self.n_base_filters * (2 ** 4), self.n_base_filters * (2 ** 3))
        self.de_conv3_1 = conv3d_block(self.n_base_filters * (2 ** 4), self.n_base_filters * (2 ** 3))
        self.de_conv3_2 = conv3d_block(self.n_base_filters * (2 ** 3), self.n_base_filters * (2 ** 3))

        self.deconv2 = trans_conv3d_block(self.n_base_filters * (2 ** 3), self.n_base_filters * (2 ** 2))
        self.de_conv2_1 = conv3d_block(self.n_base_filters * (2 ** 3), self.n_base_filters * (2 ** 2))
        self.de_conv2_2 = conv3d_block(self.n_base_filters * (2 ** 2), self.n_base_filters * (2 ** 2))

        self.deconv3 = trans_conv3d_block(self.n_base_filters * (2 ** 2), self.n_base_filters * (2 ** 1))
        self.de_conv1_1 = conv3d_block(self.n_base_filters * (2 ** 2), self.n_base_filters * (2 ** 1))
        self.de_conv1_2 = conv3d_block(self.n_base_filters * (2 ** 1), self.n_base_filters * (2 ** 1))

        ## output
        self.final_conv = conv3d_block(self.n_base_filters * (2 ** 1), self.n_base_filters)
        self.output = nn.Conv3d(self.n_base_filters, self.n_labels, kernel_size=3, stride=1, padding=1)


    def forward(self, x):

        # Encode
        #        print("x",x.shape)
        env1_1 = self.en_conv1_1(x)
        env1_2 = self.en_conv1_2(env1_1)
        pool1 = self.pool1(env1_2)
        #        print("pool1",pool1.shape)

        env2_1 = self.en_conv2_1(pool1)
        env2_2 = self.en_conv2_2(env2_1)
        pool2 = self.pool2(env2_2)
        #        print("pool2",pool2.shape)

        env3_1 = self.en_conv3_1(pool2)
        env3_2 = self.en_conv3_2(env3_1)
        pool3 = self.pool3(env3_2)
        #        print("pool3",pool3.shape)

        env4_1 = self.en_conv4_1(pool3)
        env4_2 = self.en_conv4_2(env4_1)
        #        print("env4_2",env4_2.shape)

        # Decode
        de1 = self.deconv1(env4_2)
        #        print("de1",de1.shape)

        ## dim = 0 일 거 같은데 깃에는 1로 되어 있음
        concat1 = torch.cat([de1, env3_2], dim=1)
        #        print("concat1",concat1.shape)

        de_conv3_1 = self.de_conv3_1(concat1)
        de_conv3_2 = self.de_conv3_2(de_conv3_1)

        de2 = self.deconv2(de_conv3_2)
        concat2 = torch.cat([de2, env2_2], dim=1)
        #        print("concat2",concat2.shape)

        de_conv2_1 = self.de_conv2_1(concat2)
        de_conv2_2 = self.de_conv2_2(de_conv2_1)

        de3 = self.deconv3(de_conv2_2)
        concat3 = torch.cat([de3, env1_2], dim=1)
        #        print("concat3",concat3.shape)

        de_conv1_1 = self.de_conv1_1(concat3)
        de_conv1_2 = self.de_conv1_2(de_conv1_1)

        final_conv = self.final_conv(de_conv1_2)
        #        print("final_conv",final_conv.shape)

        output = self.output(final_conv)
        #print("output", output.shape)

        #         output = self.softmax(output)
        #         print("output",output.shape)

        return output

