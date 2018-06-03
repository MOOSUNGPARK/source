import numpy as np
import tensorflow as tf
import Portpolio.Object_detection.Yolo_mobilenet.model.config as cfg
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers import *

class Mobilenet(object):
    def __init__(self, is_training=True):
        self.classes = cfg.CLASSES
        self.num_class = len(self.classes)
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.boxes_per_cell = cfg.BOXES_PER_CELL
        self.output_size = self.cell_size * self.cell_size * (self.num_class + self.boxes_per_cell * 5)
        self.scale = 1.0 * self.image_size / self.cell_size
        self.boundary1 = self.cell_size * self.cell_size * self.num_class
        self.boundary2 = self.boundary1 + self.cell_size * self.cell_size * self.boxes_per_cell

        self.object_scale = cfg.OBJECT_SCALE
        self.noobject_scale = cfg.NOOBJECT_SCALE
        self.class_scale = cfg.CLASS_SCALE
        self.coord_scale = cfg.COORD_SCALE

        self.learning_rate = cfg.LEARNING_RATE
        self.width_multiplier = cfg.WIDTH_MULTIPLIER
        self.batch_size = cfg.BATCH_SIZE

        self.offset = np.transpose(np.reshape(np.array(
            [np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),
            (self.boxes_per_cell, self.cell_size, self.cell_size)), (1, 2, 0))
        self.images = tf.placeholder(tf.float32, shape=[None, self.image_size, self.image_size, 3], name='images')
        self.logits = self._build_net()

        if is_training:
            self.labels = tf.placeholder(tf.float32, [None, self.cell_size, self.cell_size, 5 + self.num_class])
            self.loss_layer(self.logits, self.labels)
            self.total_loss = tf.losses.get_total_loss()
            tf.summary.scalar('total_loss', self.total_loss)

    def _build_net(self):

        with tf.variable_scope('Mobilenet'):

            def depthwise_convlayer(name, l, num_filter, width_multiplier, stride):
                multiplied_filter = int(num_filter * width_multiplier)
                # Depthwise
                l = slim.separable_convolution2d(l,
                                                 num_outputs = None,
                                                 kernel_size = 3,
                                                 depth_multiplier = 1,
                                                 stride = stride,
                                                 activation_fn = self.swish,
                                                 normalizer_fn = slim.batch_norm,
                                                 weights_initializer = xavier_initializer(),
                                                 biases_initializer = tf.zeros_initializer(),
                                                 scope = name + '_depthwise'
                                                 )
                # Pointwise
                l = slim.conv2d(l,
                                num_outputs = multiplied_filter,
                                kernel_size = 1,
                                activation_fn = self.swish,
                                normalizer_fn = slim.batch_norm,
                                weights_initializer = xavier_initializer(),
                                biases_initializer = tf.zeros_initializer(),
                                scope = name + '_pointwise'
                                )
                return l

            def convlayer(name, l, num_filter, width_multiplier, stride) :

                multiplied_filter = int(num_filter * width_multiplier)

                l = slim.conv2d(l,
                                num_outputs = multiplied_filter,
                                kernel_size = 3,
                                stride = stride,
                                activation_fn = self.swish,
                                normalizer_fn = slim.batch_norm,
                                weights_initializer = xavier_initializer(),
                                biases_initializer = tf.zeros_initializer(),
                                scope = name
                                )
                return l

            def fclayer(name, l, num_input, num_output):
                w = tf.get_variable(name=name + '_w', shape=[num_input, num_output], dtype=tf.float32,
                                    initializer=variance_scaling_initializer())
                b = tf.Variable(tf.constant(value=0., shape=[num_output], name=name + '_b'))
                l = tf.matmul(l, w) + b

                return l

            # print('images', self.images)
            l = convlayer('conv0', self.images, 32, self.width_multiplier, 2)
            # print('0', l)
            l = depthwise_convlayer('conv_dw1', l, 32, self.width_multiplier, 1)
            # print('1', l)
            l = depthwise_convlayer('conv_dw2', l, 64, self.width_multiplier, 2)
            # print('2', l)
            l = depthwise_convlayer('conv_dw3', l, 64, self.width_multiplier, 1)
            # print('3', l)
            l = depthwise_convlayer('conv_dw4', l, 128, self.width_multiplier, 2)
            # print('4', l)
            l = depthwise_convlayer('conv_dw5', l, 128, self.width_multiplier, 1)
            # print('5', l)
            l = depthwise_convlayer('conv_dw6', l, 256, self.width_multiplier, 2)
            # print('6', l)
            l = depthwise_convlayer('conv_dw7', l, 256, self.width_multiplier, 1)
            # print('7', l)
            l = depthwise_convlayer('conv_dw8', l, 512, self.width_multiplier, 1)
            # print('8', l)
            l = depthwise_convlayer('conv_dw9', l, 512, self.width_multiplier, 1)
            # print('9', l)
            l = depthwise_convlayer('conv_dw10', l, 512, self.width_multiplier, 1)
            # print('10', l)
            l = depthwise_convlayer('conv_dw11', l, 512, self.width_multiplier, 1)
            # print('11', l)
            l = depthwise_convlayer('conv_dw12', l, 512, self.width_multiplier, 1)
            # print('12', l)
            l = depthwise_convlayer('conv_dw13', l, 1024, self.width_multiplier, 2)
            # print('13', l)
            l = avg_pool2d(l, kernel_size=7, stride=1, padding='VALID', scope='avg_pool14')
            # print('14', l)
            l = tf.reshape(l, shape=[-1, int(1 * 1 * 1024 * self.width_multiplier)])
            # print('15', l)
            l = fclayer('fc15', l, int(1 * 1 * 1024 * self.width_multiplier), self.output_size)
            # print('16', l)
            return l

    def calc_iou(self, boxes1, boxes2, scope='iou'):
        """calculate ious
        Args:
          boxes1: 4-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
          boxes2: 1-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4] ===> (x_center, y_center, w, h)
        Return:
          iou: 3-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        """
        with tf.variable_scope(scope):
            boxes1 = tf.stack([boxes1[:, :, :, :, 0] - boxes1[:, :, :, :, 2] / 2.0,
                               boxes1[:, :, :, :, 1] - boxes1[:, :, :, :, 3] / 2.0,
                               boxes1[:, :, :, :, 0] + boxes1[:, :, :, :, 2] / 2.0,
                               boxes1[:, :, :, :, 1] + boxes1[:, :, :, :, 3] / 2.0])
            boxes1 = tf.transpose(boxes1, [1, 2, 3, 4, 0])

            boxes2 = tf.stack([boxes2[:, :, :, :, 0] - boxes2[:, :, :, :, 2] / 2.0,
                               boxes2[:, :, :, :, 1] - boxes2[:, :, :, :, 3] / 2.0,
                               boxes2[:, :, :, :, 0] + boxes2[:, :, :, :, 2] / 2.0,
                               boxes2[:, :, :, :, 1] + boxes2[:, :, :, :, 3] / 2.0])
            boxes2 = tf.transpose(boxes2, [1, 2, 3, 4, 0])

            # calculate the left up point & right down point
            lu = tf.maximum(boxes1[:, :, :, :, :2], boxes2[:, :, :, :, :2])
            rd = tf.minimum(boxes1[:, :, :, :, 2:], boxes2[:, :, :, :, 2:])

            # intersection
            intersection = tf.maximum(0.0, rd - lu)
            inter_square = intersection[:, :, :, :, 0] * intersection[:, :, :, :, 1]

            # calculate the boxs1 square and boxs2 square
            square1 = (boxes1[:, :, :, :, 2] - boxes1[:, :, :, :, 0]) * \
                (boxes1[:, :, :, :, 3] - boxes1[:, :, :, :, 1])
            square2 = (boxes2[:, :, :, :, 2] - boxes2[:, :, :, :, 0]) * \
                (boxes2[:, :, :, :, 3] - boxes2[:, :, :, :, 1])

            union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

        return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)

    def loss_layer(self, predicts, labels, scope='loss_layer'):
        with tf.variable_scope(scope):
            predict_classes = tf.reshape(predicts[:, :self.boundary1], [self.batch_size, self.cell_size, self.cell_size, self.num_class])
            predict_scales = tf.reshape(predicts[:, self.boundary1:self.boundary2], [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell])
            predict_boxes = tf.reshape(predicts[:, self.boundary2:], [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell, 4])

            response = tf.reshape(labels[:, :, :, 0], [self.batch_size, self.cell_size, self.cell_size, 1])
            boxes = tf.reshape(labels[:, :, :, 1:5], [self.batch_size, self.cell_size, self.cell_size, 1, 4])
            boxes = tf.tile(boxes, [1, 1, 1, self.boxes_per_cell, 1]) / self.image_size
            classes = labels[:, :, :, 5:]

            offset = tf.constant(self.offset, dtype=tf.float32)
            offset = tf.reshape(offset, [1, self.cell_size, self.cell_size, self.boxes_per_cell])
            offset = tf.tile(offset, [self.batch_size, 1, 1, 1])
            predict_boxes_tran = tf.stack([(predict_boxes[:, :, :, :, 0] + offset) / self.cell_size,
                                           (predict_boxes[:, :, :, :, 1] + tf.transpose(offset, (0, 2, 1, 3))) / self.cell_size,
                                           tf.square(predict_boxes[:, :, :, :, 2]),
                                           tf.square(predict_boxes[:, :, :, :, 3])])
            predict_boxes_tran = tf.transpose(predict_boxes_tran, [1, 2, 3, 4, 0])

            iou_predict_truth = self.calc_iou(predict_boxes_tran, boxes)

            # calculate I tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
            object_mask = tf.reduce_max(iou_predict_truth, 3, keep_dims=True)
            object_mask = tf.cast((iou_predict_truth >= object_mask), tf.float32) * response

            # calculate no_I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
            noobject_mask = tf.ones_like(object_mask, dtype=tf.float32) - object_mask

            boxes_tran = tf.stack([boxes[:, :, :, :, 0] * self.cell_size - offset,
                                   boxes[:, :, :, :, 1] * self.cell_size - tf.transpose(offset, (0, 2, 1, 3)),
                                   tf.sqrt(boxes[:, :, :, :, 2]),
                                   tf.sqrt(boxes[:, :, :, :, 3])])
            boxes_tran = tf.transpose(boxes_tran, [1, 2, 3, 4, 0])

            # class_loss
            class_delta = response * (predict_classes - classes)
            class_loss = tf.reduce_mean(tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]), name='class_loss') * self.class_scale

            # object_loss
            object_delta = object_mask * (predict_scales - iou_predict_truth)
            object_loss = tf.reduce_mean(tf.reduce_sum(tf.square(object_delta), axis=[1, 2, 3]), name='object_loss') * self.object_scale

            # noobject_loss
            noobject_delta = noobject_mask * predict_scales
            noobject_loss = tf.reduce_mean(tf.reduce_sum(tf.square(noobject_delta), axis=[1, 2, 3]), name='noobject_loss') * self.noobject_scale

            # coord_loss
            coord_mask = tf.expand_dims(object_mask, 4)
            boxes_delta = coord_mask * (predict_boxes - boxes_tran)
            coord_loss = tf.reduce_mean(tf.reduce_sum(tf.square(boxes_delta), axis=[1, 2, 3, 4]), name='coord_loss') * self.coord_scale

            tf.losses.add_loss(class_loss)
            tf.losses.add_loss(object_loss)
            tf.losses.add_loss(noobject_loss)
            tf.losses.add_loss(coord_loss)

            tf.summary.scalar('class_loss', class_loss)
            tf.summary.scalar('object_loss', object_loss)
            tf.summary.scalar('noobject_loss', noobject_loss)
            tf.summary.scalar('coord_loss', coord_loss)

            tf.summary.histogram('boxes_delta_x', boxes_delta[:, :, :, :, 0])
            tf.summary.histogram('boxes_delta_y', boxes_delta[:, :, :, :, 1])
            tf.summary.histogram('boxes_delta_w', boxes_delta[:, :, :, :, 2])
            tf.summary.histogram('boxes_delta_h', boxes_delta[:, :, :, :, 3])
            tf.summary.histogram('iou', iou_predict_truth)

    def swish(self, x):
        return x * tf.nn.sigmoid(x, 'swish')