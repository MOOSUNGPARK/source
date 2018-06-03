import tensorflow as tf
import datetime
import os
# import argparse
import Portpolio.Object_detection.Yolo_mobilenet.model.config as cfg
from Portpolio.Object_detection.Yolo_mobilenet.model.mobilenet_new import Mobilenet
from Portpolio.Object_detection.Yolo_mobilenet.utils.timer import Timer
from Portpolio.Object_detection.Yolo_mobilenet.utils.pascal import pascal_voc

class Train_model(object):
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.weights_file = cfg.WEIGHTS_FILE
        self.max_iter = cfg.MAX_ITER
        self.initial_learning_rate = cfg.LEARNING_RATE
        self.decay_steps = cfg.DECAY_STEPS
        self.decay_rate = cfg.DECAY_RATE
        self.staircase = cfg.STAIRCASE
        self.summary_iter = cfg.SUMMARY_ITER
        self.save_iter = cfg.SAVE_ITER
        self.output_dir = os.path.join(cfg.OUTPUT_DIR, datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.save_cfg()

        self.restore = cfg.RESTORE
        # self.restorer = tf.train.Saver(tf.global_variables(), max_to_keep=None)
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
        self.ckpt_file = os.path.join(self.output_dir, 'save.ckpt')
        self.summary_op = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.output_dir, flush_secs=60)

        self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        self.learning_rate = tf.train.exponential_decay(self.initial_learning_rate,
                                                        self.global_step,
                                                        self.decay_steps,
                                                        self.decay_rate,
                                                        self.staircase,
                                                        name='learning_rate')
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.model.total_loss,
                                                                             global_step=self.global_step)
        self.ema = tf.train.ExponentialMovingAverage(decay=0.9999)
        self.averages_op = self.ema.apply(tf.trainable_variables())
        with tf.control_dependencies([self.optimizer]):
            self.train_op = tf.group(self.averages_op)

        # gpu_options = tf.GPUOptions(allow_growth=True)
        gpu_options = tf.GPUOptions()
        config = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        if self.restore :
            # print('Restoring weigths from : ' + self.weights_file)
            # self.restorer = tf.train.import_meta_graph(self.weights_file + '.meta')
            # # self.restorer.restore(self.sess, self.weights_file)
            # self.restorer.restore(self.sess, tf.train.latest_checkpoint(
            #     'C:\\python\\source\\Portpolio\\Object_detection\\Yolo_mobilenet\\data\\train\\weights'))
            print('Restoring weigths from : ' + self.weights_file)
            # self.restorer = tf.train.import_meta_graph(self.weights_file + '.meta')
            # self.restorer.restore(self.sess, self.weights_file)
            self.saver.restore(self.sess, tf.train.latest_checkpoint(cfg.WEIGHTS_DIR))


        self.writer.add_graph(self.sess.graph)

    def train(self):
        train_timer = Timer()
        load_timer = Timer()

        for step in range(1, self.max_iter + 1):
            load_timer.tic()
            images, labels = self.data.get()
            load_timer.toc()
            feed_dict = {self.model.images : images, self.model.labels : labels}

            if step % self.summary_iter == 0:
                if step % (self.summary_iter * 10) == 0:
                    train_timer.tic()
                    summary_str, loss, _ = self.sess.run([self.summary_op, self.model.total_loss, self.train_op],
                                                         feed_dict=feed_dict)
                    train_timer.toc()

                    log_str = ('{} Epoch: {}, Step: {}, Learning rate: {}, Loss: {:5.3f}\n\t'
                               'Speed: {:.3f}s/iter, Load: {:.3f}s/iter, Remain: {}').format(
                        datetime.datetime.now().strftime('%m/%d %H:%M:%S'),
                        self.data.epoch,
                        int(step),
                        round(self.learning_rate.eval(session=self.sess), 10),
                        loss,
                        train_timer.average_time,
                        load_timer.average_time,
                        train_timer.remain(step, self.max_iter))
                    print(log_str)

                else:
                    train_timer.tic()
                    summary_str, _ = self.sess.run([self.summary_op, self.train_op], feed_dict=feed_dict)
                    train_timer.toc()
                # print(self.model.logits.eval(session=self.sess))
                # print(self.model.labels.eval(session=self.sess))
                self.writer.add_summary(summary_str, step)

            else:
                train_timer.tic()
                self.sess.run(self.train_op, feed_dict=feed_dict)
                train_timer.toc()

            if step % self.save_iter == 0:
                print('\n{} {} Epoch\'s Saving checkpoint file to: {}\n'.format(
                    datetime.datetime.now().strftime('%m/%d %H:%M:%S'),
                    self.data.epoch,
                    self.output_dir))
                self.saver.save(self.sess, self.ckpt_file, global_step=self.global_step)

    def save_cfg(self):

        with open(os.path.join(self.output_dir, 'config.txt'), 'w') as f:
            cfg_dict = cfg.__dict__
            for key in sorted(cfg_dict.keys()):
                if key[0].isupper():
                    cfg_str = '{}: {}\n'.format(key, cfg_dict[key])
                    f.write(cfg_str)

# def update_config_paths(data_dir, weights_file):
#     cfg.TRAIN_DATA_PATH = data_dir
#     cfg.PASCAL_PATH = os.path.join(data_dir, 'pascal_voc')
#     cfg.CACHE_PATH = os.path.join(cfg.PASCAL_PATH, 'cache')
#     cfg.OUTPUT_DIR = os.path.join(cfg.PASCAL_PATH, 'output')
#     cfg.WEIGHTS_DIR = os.path.join(cfg.PASCAL_PATH, 'weights')
#     cfg.WEIGHTS_FILE = os.path.join(cfg.WEIGHTS_DIR, weights_file)

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', default="YOLO_small.ckpt", type=str)
    # parser.add_argument('--data_dir', default=cfg.TRAIN_DATA_PATH, type=str)
    # parser.add_argument('--threshold', default=0.2, type=float)
    # parser.add_argument('--iou_threshold', default=0.5, type=float)
    # parser.add_argument('--gpu', default='', type=str)
    # args = parser.parse_args()

    # if args.gpu is not None:
    #     cfg.GPU = args.gpu

    # if args.data_dir != cfg.TRAIN_DATA_PATH:
    #     update_config_paths(args.data_dir, args.weights)

    # os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU

    model = Mobilenet(is_training=True)
    pascal = pascal_voc('train')

    train_model = Train_model(model, pascal)

    print('Start training')
    train_model.train()
    print('Training ended')

if __name__ == '__main__':
    main()
