import time
import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='6'
import numpy as np
import tensorflow as tf
from Densenet import densenet
import sys
sys.path.append('/home/mspark/practice/Cifar10_classify')
import config as cfg
import data.cifar10_pickle as cifar10


class Train_model():
    def __init__(self, model):
        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(gpu_options=gpu_options)
        with tf.Session(config=config) as sess:
            self.sess = sess

        self.model = model
        self.epochs = cfg.EPOCHS
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

        if not os.path.exists(cfg.CKPT_DIR_PATH):
            os.makedirs(cfg.CKPT_DIR_PATH)

        if cfg.RESTORE :
            print('Restoring weigths from : ' + cfg.CKPT_FILE)
            self.saver.restore(self.sess, tf.train.latest_checkpoint(cfg.CKPT_DIR_PATH))

    def train(self):

        train_tot_data, train_tot_label = cifar10.load_training_data()
        train_x, train_y = train_tot_data[0: 45000], train_tot_label[0: 45000]
        validation_x, validation_y = train_tot_data[45000:], train_tot_label[45000:]

        print('Learning Started!')
        stime = time.time()

        for epoch in range(self.epochs):
            epoch_stime = time.time()
            train_accuracy = []
            train_loss = 0.
            validation_accuracy = []

            # train
            for start_idx in range(0, len(train_x), cfg.BATCH_SIZE):
                train_x_batch = train_x[start_idx:start_idx+cfg.BATCH_SIZE]
                train_y_batch = train_y[start_idx:start_idx+cfg.BATCH_SIZE]

                feed_dict = {self.model.X: train_x_batch, self.model.Y: train_y_batch, self.model.training: True}

                train_acc, loss = self.sess.run([self.model.accuracy, self.model.train_op], feed_dict=feed_dict)
                train_loss += loss / cfg.BATCH_SIZE

                train_accuracy.append(train_acc)

            # validation
            for start_idx in range(0, len(validation_x), cfg.BATCH_SIZE):
                validation_x_batch = validation_x[start_idx:start_idx+cfg.BATCH_SIZE]
                validation_y_batch = validation_y[start_idx:start_idx+cfg.BATCH_SIZE]

                feed_dict = {self.model.X: validation_x_batch, self.model.Y: validation_y_batch, self.model.training: False}

                validation_acc = self.sess.run([self.model.accuracy], feed_dict=feed_dict)
                validation_accuracy.append(validation_acc)


            # save current state
            if (epoch + 1) % (cfg.SAVE_EPOCHS) == 0:
                self.saver.save(self.sess, cfg.CKPT_FILE, global_step=self.model.global_step)

            epoch_etime = time.time()

            print('epoch :', epoch + 1, ', global_step :', self.sess.run(self.model.global_step),
                  ', loss :', train_loss, ', train_accuracy :', np.mean(np.array(train_accuracy)),
                  ', validation_accuracy :', np.mean(np.array(validation_accuracy)),
                  ', time :', round(epoch_etime-epoch_stime, 6))

        print('Learning Finished')
        etime = time.time()

        print('consumption time : ', round(etime-stime, 6))

def main():
    model = densenet(name='resnet', label_cnt=cfg.LABEL_CNT)
    train_model = Train_model(model)
    train_model.train()


    if cfg.EXPORT:
        builder = tf.saved_model.builder.SavedModelBuilder(cfg.CKPT_DIR_PATH + '\saved')
        builder.add_meta_graph_and_variables(train_model.sess,
                                             [tf.saved_model.tag_constants.TRAINING],
                                             signature_def_map=None,
                                             assets_collection=None)
        builder.save()

if __name__ == '__main__':
    main()
