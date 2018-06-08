"""
FCN(Fully Convolutional Network)의 종류이며 다양하게 쓰이고 있는 U-Net(https://arxiv.org/abs/1505.04597)을 기반으로 한 네트워크입니다.
FCN과 U-Net에서 Upsampling 과정에서 사용한 기존의 deconvolution(Transpose Convolution) 대신 Cycle GAN에서 사용한 Resize Convolution(https://distill.pub/2016/deconv-checkerboard/)을 사용하였으며
배치정규화와 그룹정규화(https://arxiv.org/pdf/1803.08494.pdf)를 통해 정규화를 합니다.
4번의 Downsampling과 4번의 Upsampling을 진행하며 이를 통해 이미지를 학습하고 라벨이미지에 가깝게 결과물을 출력하게 합니다.

옵션으로 다양한 정규화 방법과 활성화 함수들을 사용 할 수 있으며 모델 구성에 사용된 메소드는 utils.py에 구현되어 있습니다.

보통 이미지 세그먼테이션에서 실제 이미지를 Groundtruth라고 하고 예측 이미지를 Predicted 라고 합니다.

신경망의 기본 구조는 아래와 같습니다.

- 크기 다운샘플링-업샘플링을 통해 결과 이미지를 얻습니다.

- 다운샘플링 한 층은 Convolution-Activation-Convolution-Activation-Pool 로 이루어져 있으며 일반적으로 3x3 필터를 사용합니다.

- 업샘플링 한 층은 Upsample-Activation-Convolution-Activation-Convolution-Activation 로 이루어져 있으며 마찬자기로 3x3필터를 사용합니다.

- 업샘플링 과정에서 중간중간 Concatenation을 통해 다운샘플링 과정의 Feature map을 전달해줌으로써 데이터손실을 방지합니다.

- 네트워크 자체와 데이터가 매우 무겁기 때문에 Fully Connected Layer는 사용하지 않습니다. 1x1 필터를 사용하는 1x1 Convolution으로 FC Layer가 대체 가능합니다.

"""


import tensorflow as tf
import Portpolio.brain_tumor.unet.utils as utils


class Model:

    def __init__(self, loss='dice', upsampling_mode='resize', normalization_mode='Batch', groupnormalization_n=2, downsampling_option='neighbor',
                 model_root_channel=32, img_size=256, batch_size=20, n_channel=1, n_class=2, activation='relu', channel_mode='NHWC'):
        # feed dict로 넘어오지 않고 다른 .py 파일에서 넘겨받아오는 옵션값들입니다. 배치사이즈와 정규화방법, 업샘플링방법, 초기채널수 등이 있습니다.
        self.batch_size = batch_size
        self.normalization_mode = normalization_mode
        self.model_channel = model_root_channel
        self.upsampling_mode = upsampling_mode
        self.loss_mode = loss
        self.group_n = groupnormalization_n
        self.activation = activation
        self.channel_mode = channel_mode
        self.downsampling_opt = downsampling_option
        self.img_size = img_size

        # 모델에 사용하는 변수들로 feed dict로 넘어오는 것 들은 tf.placeholder로 Tensor 공간을 만들어주어야 합니다.
        self.drop_rate = tf.placeholder(tf.float32)
        self.training = tf.placeholder(tf.bool)
        if self.channel_mode == 'NHWC':
            self.X = tf.placeholder(tf.float32, [self.batch_size, img_size, img_size, n_channel], name='X')
            self.Y = tf.placeholder(tf.float32, [self.batch_size, img_size, img_size, n_class], name='Y')
        elif self.channel_mode == 'NCHW':
            self.X = tf.placeholder(tf.float32, [self.batch_size, n_channel, img_size, img_size], name='X')
            self.Y = tf.placeholder(tf.float32, [self.batch_size, n_class, img_size, img_size], name='Y')





        # 네트워크의 출력 결과물입니다.
        self.logits = self.neural_net()

        # 출력 결과물을 pixelwise softmax를 통해 픽셀별 predict probability로 활성화시킵니다.
        self.pred = tf.nn.softmax(logits=self.logits)

        # 활성화 시킨 probability map을 split 하여 foreground와 background로 분리합니다.
        self.foreground_predicted, self.background_predicted = tf.split(self.pred, [1, 1], 3)

        # 라벨이미지 역시 foreground와 background로 분리합니다
        self.foreground_truth, self.background_truth = tf.split(self.Y, [1, 1], 3)

        # loss 함수는 Dice Loss로 진행합니다. utils에 여러종류의 Loss function이 구현되어있습니다.
        # 옵션으로 Dice Loss, Focal Loss, Cross Entropy Loss, Huber Loss 등을 사용 할 수 있습니다.
        if self.loss_mode == 'dice':
            self.loss = utils.dice_loss(output=self.foreground_predicted, target=self.foreground_truth)
        elif self.loss_mode == 'focal':
            self.loss = utils.focal_loss(output=self.foreground_predicted, target=self.foreground_truth)
        elif self.loss_mode == 'cross_entropy':
            self.loss = utils.cross_entropy(output=self.foreground_predicted, target=self.foreground_truth)
        elif self.loss_mode == 'dice_sum':
            self.loss = utils.dice_loss_sum(output=self.foreground_predicted, target=self.foreground_truth)

        # segmentation accuracy는 보통 IoU라고 하는 Intersection over Union을 사용합니다.

        self.results = list(utils.iou_coe(output=self.foreground_predicted, target=self.foreground_truth))

    def neural_net(self):
        depth = 4
        down_conv = [0] * depth
        down_pool = [0] * depth
        up_conv = [0] * depth
        up_pool = [0] * depth

        # 다운샘플링
        with tf.name_scope('down'):
            next_input = self.X
            channel_n = self.model_channel
            pool_size = self.img_size
            for i in range(depth):
                pool_size //= 2

                down_conv[i] = utils.conv2D('conv' + str(i) + '_1', next_input, channel_n, [3, 3], [1, 1], 'same')
                down_conv[i] = utils.Normalization(down_conv[i], self.normalization_mode, self.training, 'c' + str(i) + '_1', G=self.group_n, channel_mode=self.channel_mode)  # 옵션으로 Batch Normalization과 Group Normalization을 사용 할 수 있습니다.
                down_conv[i] = utils.activation('act' + str(i) + '_1', down_conv[i], type='lrelu')  # 옵션으로 Relu, Leaky Relu, Selu, Elu 등을 사용 할 수 있습니다.
                down_conv[i] = utils.conv2D('conv' + str(i) + '_2', down_conv[i], channel_n, [3, 3], [1, 1], 'same')
                down_conv[i] = utils.Normalization(down_conv[i], self.normalization_mode, self.training, 'c' + str(i) + '_2', G=self.group_n, channel_mode=self.channel_mode)
                down_conv[i] = utils.activation('act' + str(i) + '_2', down_conv[i], type='lrelu')

                if self.downsampling_opt == 'linear':
                    down_pool[i] = tf.image.resize_bilinear(images=down_conv[i], size=[pool_size, pool_size],
                                                            name='down' + str(i) + '_resizing')
                    down_pool[i] = utils.conv2D(name='pool' + str(i) + '_conv', inputs=down_pool[i], filters=channel_n,
                                                kernel_size=[1, 1], strides=[1, 1], padding='same')
                    down_conv[i] = utils.Normalization(down_conv[i], self.normalization_mode, self.training,
                                                       'pool' + str(i) + '_c', G=self.group_n,
                                                       channel_mode=self.channel_mode)
                    down_conv[i] = utils.activation('pool' + str(i) + '_act', down_conv[i], type='lrelu')
                elif self.downsampling_opt == 'neighbor':
                    down_pool[i] = tf.image.resize_nearest_neighbor(images=down_conv[i], size=[pool_size, pool_size], name='down' + str(i) + '_resizing')
                    down_pool[i] = utils.conv2D(name='pool' + str(i) + '_conv', inputs=down_pool[i], filters=channel_n,
                                                kernel_size=[1, 1], strides=[1, 1], padding='same')
                    down_conv[i] = utils.Normalization(down_conv[i], self.normalization_mode, self.training,
                                                       'pool' + str(i) + '_c', G=self.group_n,
                                                       channel_mode=self.channel_mode)
                    down_conv[i] = utils.activation('pool' + str(i) + '_act', down_conv[i], type='lrelu')
                elif self.downsampling_opt == 'maxpool':
                    down_pool[i] = utils.maxpool('pool' + str(i) + '_conv', down_conv[i], [2, 2], [2, 2], 'same')
                else:
                    raise Exception

                channel_n *= 2
                next_input = down_pool[i]

            same_conv = utils.conv2D('conv5_1', next_input, channel_n, [3, 3], [1, 1], 'same')
            same_conv = utils.Normalization(same_conv, self.normalization_mode, self.training, 'c5_1', G=self.group_n, channel_mode=self.channel_mode)
            same_conv = utils.activation('act5_1', same_conv, type='lrelu')
            same_conv = utils.conv2D('conv5_2', same_conv, channel_n, [1, 1], [1, 1], 'same')
            same_conv = utils.Normalization(same_conv, self.normalization_mode, self.training, 'c5_2', G=self.group_n, channel_mode=self.channel_mode)
            same_conv = utils.activation('act5_2', same_conv, type='lrelu')

        # 업샘플링
        with tf.name_scope('up'):
            next_input = same_conv
            for i in reversed(range(depth)):
                pool_size *= 2

                if self.upsampling_mode == 'resize':  # 옵션으로 reconv convolution과 deconvolution을 사용 할 수 있습니다.
                    up_pool[i] = utils.re_conv2D(name='reconv' + str(i), inputs=next_input, output_shape=[-1, pool_size, pool_size, channel_n // 2])
                elif self.upsampling_mode == 'transpose':   # unbalanced u-net의 경우 오류 발생 중.
                    up_pool[i] = utils.deconv2D('deconv' + str(i), next_input, [3, 3, channel_n // 2, channel_n], [-1, pool_size, pool_size, channel_n // 2], [1, 2, 2, 1], 'SAME')
                    up_pool[i] = tf.reshape(up_pool[i], shape=[-1, pool_size, pool_size, channel_n // 2])
                print('up_pool[{0}]: {1}, down_conv[{0}]: {2}'.format(str(i), up_pool[i].shape, down_conv[i].shape))
                up_pool[i] = utils.Normalization(up_pool[i], self.normalization_mode, self.training, 'r' + str(i), G=self.group_n, channel_mode=self.channel_mode)
                up_pool[i] = utils.activation('deact' + str(i), up_pool[i], type='lrelu')
                up_pool[i] = utils.concat('concat' + str(i), [up_pool[i], down_conv[i]], 3)

                channel_n //= 2
                up_conv[i] = utils.conv2D('uconv' + str(i) + '_1', up_pool[i], channel_n, [3, 3], [1, 1], 'same')
                up_conv[i] = utils.Normalization(up_conv[i], self.normalization_mode, self.training, 'uc' + str(i) + '_1', G=self.group_n, channel_mode=self.channel_mode)
                up_conv[i] = utils.activation('uact' + str(i) + '_1', up_conv[i], type='lrelu')
                up_conv[i] = utils.conv2D('uconv' + str(i) + '_2', up_conv[i], channel_n, [3, 3], [1, 1], 'same')
                up_conv[i] = utils.Normalization(up_conv[i], self.normalization_mode, self.training, 'uc' + str(i) + '_2', G=self.group_n, channel_mode=self.channel_mode)
                up_conv[i] = utils.activation('uact' + str(i) + '_2', up_conv[i], type='lrelu')

                next_input = up_conv[i]

            out_seg = utils.conv2D('output_seq', next_input, 2, [1, 1], [1, 1], 'same')

        return out_seg
    #
    #
    #
    #
    # def neural_net(self):
    #     # 다운샘플링
    #     with tf.name_scope('down'):
    #         channel_n = self.model_channel
    #         conv1 = utils.conv2D('conv1_1', self.X, channel_n, [3, 3], [1, 1], 'same')
    #         conv1 = utils.Normalization(conv1, self.normalization_mode, self.training, 'c1_1', G=self.group_n, channel_mode=self.channel_mode)  # 옵션으로 Batch Normalization과 Group Normalization을 사용 할 수 있습니다.
    #         conv1 = utils.activation('act1_1', conv1, type='lrelu')  # 옵션으로 Relu, Leaky Relu, Selu, Elu 등을 사용 할 수 있습니다.
    #         conv1 = utils.conv2D('conv1_2', conv1, channel_n, [3, 3], [1, 1], 'same')
    #         conv1 = utils.Normalization(conv1, self.normalization_mode, self.training, 'c1_2', G=self.group_n, channel_mode=self.channel_mode)
    #         conv1 = utils.activation('act1_2', conv1, type='lrelu')
    #
    # if self.downsampling_opt == 'linear':
    #     pool1 = tf.image.resize_nearest_neighbor(images=conv1, size=[128, 128], name='down1_resizing')
    #     pool1 = utils.conv2D(name='pool1_conv', inputs=pool1, filters=channel_n, kernel_size=[1, 1], strides=[1, 1], padding='same')
    #     pool1 = utils.Normalization(pool1, self.normalization_mode, self.training, 'c1_3', G=self.group_n,channel_mode=self.channel_mode)
    #     pool1 = utils.activation('act1_3', pool1, type='lrelu')
    # elif self.downsampling_opt == 'neighbor':
    #     pool1 = tf.image.resize_bilinear(images=conv1, size=[128, 128], name='down1_resizing')
    #     pool1 = utils.conv2D(name='pool1_conv', inputs=pool1, filters=channel_n, kernel_size=[1, 1], strides=[1, 1], padding='same')
    #     pool1 = utils.Normalization(pool1, self.normalization_mode, self.training, 'c1_3', G=self.group_n, channel_mode=self.channel_mode)
    #     pool1 = utils.activation('act1_3', pool1, type='lrelu')
    # else:
    #     pool1 = utils.maxpool('pool1', conv1, [2, 2], [2, 2], 'same')
    #
    #         channel_n *= 2
    #         conv2 = utils.conv2D('conv2_1', pool1, channel_n, [3, 3], [1, 1], 'same')
    #         conv2 = utils.Normalization(conv2, self.normalization_mode, self.training, 'c2_1', G=self.group_n, channel_mode=self.channel_mode)
    #         conv2 = utils.activation('act2_1', conv2, type='lrelu')
    #         conv2 = utils.conv2D('conv2_2', conv2, channel_n, [3, 3], [1, 1], 'same')
    #         conv2 = utils.Normalization(conv2, self.normalization_mode, self.training, 'c2_2', G=self.group_n, channel_mode=self.channel_mode)
    #         conv2 = utils.activation('act2_2', conv2, type='lrelu')
    #
    #         if self.downsampling_opt == 'linear':
    #             pool2 = tf.image.resize_nearest_neighbor(images=conv2, size=[64, 64], name='down1_resizing')
    #             pool2 = utils.conv2D(name='pool2_conv', inputs=pool2, filters=channel_n, kernel_size=[1, 1], strides=[1, 1], padding='same')
    #             pool2 = utils.Normalization(pool2, self.normalization_mode, self.training, 'c2_3', G=self.group_n, channel_mode=self.channel_mode)
    #             pool2 = utils.activation('act2_3', pool2, type='lrelu')
    #         elif self.downsampling_opt == 'neighbor':
    #             pool2 = tf.image.resize_bilinear(images=conv2, size=[64, 64], name='down1_resizing')
    #             pool2 = utils.conv2D(name='pool2_conv', inputs=pool2, filters=channel_n, kernel_size=[1, 1], strides=[1, 1], padding='same')
    #             pool2 = utils.Normalization(pool2, self.normalization_mode, self.training, 'c2_3', G=self.group_n, channel_mode=self.channel_mode)
    #             pool2 = utils.activation('act2_3', pool2, type='lrelu')
    #         else:
    #             pool2 = utils.maxpool('pool2', conv2, [2, 2], [2, 2], 'same')
    #
    #         channel_n *= 2
    #         conv3 = utils.conv2D('conv3_1', pool2, channel_n, [3, 3], [1, 1], 'same')
    #         conv3 = utils.Normalization(conv3, self.normalization_mode, self.training, 'c3_1', G=self.group_n, channel_mode=self.channel_mode)
    #         conv3 = utils.activation('act3_1', conv3, type='lrelu')
    #         conv3 = utils.conv2D('conv3_2', conv3, channel_n, [3, 3], [1, 1], 'same')
    #         conv3 = utils.Normalization(conv3, self.normalization_mode, self.training, 'c3_2', G=self.group_n, channel_mode=self.channel_mode)
    #         conv3 = utils.activation('act3_2', conv3, type='lrelu')
    #
    #         if self.downsampling_opt == 'linear':
    #             pool3 = tf.image.resize_nearest_neighbor(images=conv1, size=[32, 32], name='down1_resizing')
    #             pool3 = utils.conv2D(name='pool3_conv', inputs=pool3, filters=channel_n, kernel_size=[1, 1], strides=[1, 1], padding='same')
    #             pool3 = utils.Normalization(pool3, self.normalization_mode, self.training, 'c3_3', G=self.group_n, channel_mode=self.channel_mode)
    #             pool3 = utils.activation('act3_3', pool3, type='lrelu')
    #         elif self.downsampling_opt == 'neighbor':
    #             pool3 = tf.image.resize_bilinear(images=conv1, size=[32, 32], name='down1_resizing')
    #             pool3 = utils.conv2D(name='pool3_conv', inputs=pool3, filters=channel_n, kernel_size=[1, 1], strides=[1, 1], padding='same')
    #             pool3 = utils.Normalization(pool3, self.normalization_mode, self.training, 'c3_3', G=self.group_n, channel_mode=self.channel_mode)
    #             pool3 = utils.activation('act3_3', pool3, type='lrelu')
    #         else:
    #             pool3 = utils.maxpool('pool3', conv3, [2, 2], [2, 2], 'same')
    #
    #         channel_n *= 2
    #         conv4 = utils.conv2D('conv4_1', pool3, channel_n, [3, 3], [1, 1], 'same')
    #         conv4 = utils.Normalization(conv4, self.normalization_mode, self.training, 'c4_1', G=self.group_n, channel_mode=self.channel_mode)
    #         conv4 = utils.activation('act4_1', conv4, type='lrelu')
    #         conv4 = utils.conv2D('conv4_2', conv4, channel_n, [3, 3], [1, 1], 'same')
    #         conv4 = utils.Normalization(conv4, self.normalization_mode, self.training, 'c4_2', G=self.group_n, channel_mode=self.channel_mode)
    #         conv4 = utils.activation('act4_2', conv4, type='lrelu')
    #
    #         if self.downsampling_opt == 'linear':
    #             pool4 = tf.image.resize_nearest_neighbor(images=conv4, size=[16, 16], name='down1_resizing')
    #             pool4 = utils.conv2D(name='pool4_conv', inputs=pool4, filters=channel_n, kernel_size=[1, 1], strides=[1, 1], padding='same')
    #             pool4 = utils.Normalization(pool4, self.normalization_mode, self.training, 'c4_3', G=self.group_n, channel_mode=self.channel_mode)
    #             pool4 = utils.activation('act4_3', pool4, type='lrelu')
    #         elif self.downsampling_opt == 'neighbor':
    #             pool4 = tf.image.resize_bilinear(images=conv4, size=[16, 16], name='down1_resizing')
    #             pool4 = utils.conv2D(name='pool4_conv', inputs=pool4, filters=channel_n, kernel_size=[1, 1], strides=[1, 1], padding='same')
    #             pool4 = utils.Normalization(pool4, self.normalization_mode, self.training, 'c4_3', G=self.group_n, channel_mode=self.channel_mode)
    #             pool4 = utils.activation('act4_3', pool4, type='lrelu')
    #         else:
    #             pool4 = utils.maxpool('pool4', conv4, [2, 2], [2, 2], 'same')
    #
    #         channel_n *= 2
    #         conv5 = utils.conv2D('conv5_1', pool4, channel_n, [3, 3], [1, 1], 'same')
    #         conv5 = utils.Normalization(conv5, self.normalization_mode, self.training, 'c5_1', G=self.group_n, channel_mode=self.channel_mode)
    #         conv5 = utils.activation('act5_1', conv5, type='lrelu')
    #         conv5 = utils.conv2D('conv5_2', conv5, channel_n, [3, 3], [1, 1], 'same')
    #         conv5 = utils.Normalization(conv5, self.normalization_mode, self.training, 'c5_2', G=self.group_n, channel_mode=self.channel_mode)
    #         conv5 = utils.activation('act5_2', conv5, type='lrelu')
    #
    #     # 업샘플링
    #     with tf.name_scope('up'):
    #         if self.upsampling_mode == 'resize':  # 옵션으로 reconv convolution과 deconvolution을 사용 할 수 있습니다.
    #             up4 = utils.re_conv2D(name='reconv4', inputs=conv5, output_shape=[-1, 32, 32, channel_n // 2])
    #         else:
    #             up4 = utils.deconv2D('deconv4', conv5, [3, 3, channel_n // 2, channel_n], [-1, 32, 32, channel_n // 2], [1, 2, 2, 1], 'SAME')
    #             up4 = tf.reshape(up4, shape=[-1, 32, 32, channel_n // 2])
    #
    #         up4 = utils.Normalization(up4, self.normalization_mode, self.training, 'r4', G=self.group_n, channel_mode=self.channel_mode)
    #         up4 = utils.activation('deact4', up4, type='lrelu')
    #         up4 = utils.concat('concat4', [up4, conv4], 3)
    #
    #         channel_n //= 2
    #         conv4 = utils.conv2D('uconv4_1', up4, channel_n, [3, 3], [1, 1], 'same')
    #         conv4 = utils.Normalization(conv4, self.normalization_mode, self.training, 'uc4_1', G=self.group_n, channel_mode=self.channel_mode)
    #         conv4 = utils.activation('uact4-1', conv4, type='lrelu')
    #         conv4 = utils.conv2D('uconv4_2', conv4, channel_n, [3, 3], [1, 1], 'same')
    #         conv4 = utils.Normalization(conv4, self.normalization_mode, self.training, 'uc4_2', G=self.group_n, channel_mode=self.channel_mode)
    #         conv4 = utils.activation('uact4-2', conv4, type='lrelu')
    #
    #         if self.upsampling_mode == 'resize':
    #             up3 = utils.re_conv2D(name='reconv3', inputs=conv4, output_shape=[-1, 64, 64, channel_n // 2])
    #         else:
    #             up3 = utils.deconv2D('deconv3', conv4, [3, 3, channel_n // 2, channel_n], [-1, 64, 64, channel_n // 2], [1, 2, 2, 1], 'SAME')
    #             up3 = tf.reshape(up3, shape=[-1, 64, 64, channel_n // 2])
    #
    #         up3 = utils.Normalization(up3, self.normalization_mode, self.training, 'r3', G=self.group_n, channel_mode=self.channel_mode)
    #         up3 = utils.activation('deact3', up3, type='lrelu')
    #         up3 = utils.concat('concat3', [up3, conv3], 3)
    #
    #         channel_n //= 2
    #         conv3 = utils.conv2D('uconv3_1', up3, channel_n, [3, 3], [1, 1], 'same')
    #         conv3 = utils.Normalization(conv3, self.normalization_mode, self.training, 'uc3_1', G=self.group_n, channel_mode=self.channel_mode)
    #         conv3 = utils.activation('uact3-1', conv3, type='lrelu')
    #         conv3 = utils.conv2D('uconv3_2', conv3, channel_n, [3, 3], [1, 1], 'same')
    #         conv3 = utils.Normalization(conv3, self.normalization_mode, self.training, 'uc3_2', G=self.group_n, channel_mode=self.channel_mode)
    #         conv3 = utils.activation('uact3-2', conv3, type='lrelu')
    #
    #         if self.upsampling_mode == 'resize':
    #             up2 = utils.re_conv2D(name='reconv2', inputs=conv3, output_shape=[-1, 128, 128, channel_n // 2])
    #         else:
    #             up2 = utils.deconv2D('deconv2', conv3, [3, 3, channel_n // 2, channel_n], [-1, 128, 128, channel_n // 2], [1, 2, 2, 1], 'SAME')
    #             up2 = tf.reshape(up2, shape=[-1, 128, 128, channel_n // 2])
    #
    #         up2 = utils.Normalization(up2, self.normalization_mode, self.training, 'r2', G=self.group_n, channel_mode=self.channel_mode)
    #         up2 = utils.activation('deact2', up2, type='lrelu')
    #         up2 = utils.concat('concat2', [up2, conv2], 3)
    #
    #         channel_n //= 2
    #         conv2 = utils.conv2D('uconv2_1', up2, channel_n, [3, 3], [1, 1], 'same')
    #         conv2 = utils.Normalization(conv2, self.normalization_mode, self.training, 'uc2_1', G=self.group_n, channel_mode=self.channel_mode)
    #         conv2 = utils.activation('uact2-1', conv2, type='lrelu')
    #         conv2 = utils.conv2D('uconv2_2', conv2, channel_n, [3, 3], [1, 1], 'same')
    #         conv2 = utils.Normalization(conv2, self.normalization_mode, self.training, 'uc2_2', G=self.group_n, channel_mode=self.channel_mode)
    #         conv2 = utils.activation('uact2-2', conv2, type='lrelu')
    #
    #         if self.upsampling_mode == 'resize':
    #             up1 = utils.re_conv2D(name='reconv1', inputs=conv2, output_shape=[-1, 256, 256, channel_n // 2])
    #         else:
    #             up1 = utils.deconv2D('deconv1', conv2, [3, 3, channel_n // 2, channel_n], [-1, 256, 256, channel_n // 2], [1, 2, 2, 1], 'SAME')
    #             up1 = tf.reshape(up1, shape=[-1, 256, 256, channel_n // 2])
    #
    #         up1 = utils.Normalization(up1, self.normalization_mode, self.training, 'r1', G=self.group_n, channel_mode=self.channel_mode)
    #         up1 = utils.activation('deact1', up1, type='lrelu')
    #         up1 = utils.concat('concat1', [up1, conv1], 3)
    #
    #         channel_n //= 2
    #         conv1 = utils.conv2D('uconv1_1', up1, 16, [3, 3], [1, 1], 'same')
    #         conv1 = utils.Normalization(conv1, self.normalization_mode, self.training, 'uc1_1', G=self.group_n, channel_mode=self.channel_mode)
    #         conv1 = utils.activation('uact1-1', conv1, type='lrelu')
    #         conv1 = utils.conv2D('uconv1_2', conv1, 16, [3, 3], [1, 1], 'same')
    #         conv1 = utils.Normalization(conv1, self.normalization_mode, self.training, 'uc1_2', G=self.group_n, channel_mode=self.channel_mode)
    #         conv1 = utils.activation('uact1-2', conv1, type='lrelu')
    #
    #         out_seg = utils.conv2D('uconv1', conv1, 2, [1, 1], [1, 1], 'same')
    #
    #     return out_seg
    #
