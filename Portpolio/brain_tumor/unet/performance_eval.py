import tensorflow as tf
import numpy as np


class performance:
    def __init__(self):
        # 텐서보드 스칼라 서머리용 플레이스홀더
        self.acc = tf.placeholder(tf.float32)
        self.loss = tf.placeholder(tf.float32)
        self.mean_iou = tf.placeholder(tf.float32)
        self.tot_iou = tf.placeholder(tf.float32)

        # 텐서 스칼라 값을 텐서보드에 기록합니다.
        tf.summary.scalar('Loss', self.loss)
        tf.summary.scalar('Accuracy', self.acc)
        tf.summary.scalar('Mean IoU when Correct', self.mean_iou)
        tf.summary.scalar('Total IoU', self.tot_iou)