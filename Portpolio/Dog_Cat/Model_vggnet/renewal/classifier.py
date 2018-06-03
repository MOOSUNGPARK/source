import tensorflow as tf
import numpy as np
from Portpolio.Dog_Cat.Model_vggnet.renewal.util import read_data
import Portpolio.Dog_Cat.Model_vggnet.renewal.config as cfg
from Portpolio.Dog_Cat.Model_vggnet.renewal.vggnet import Vggnet

class Classifier():
    def __init__(self, model):
        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(gpu_options=gpu_options)
        with tf.Session(config=config) as sess:
            self.sess = sess

        self.model = model
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
        print('Restoring weigths from : ' + cfg.CKPT_DIR_PATH)
        self.saver.restore(self.sess, tf.train.latest_checkpoint(cfg.CKPT_DIR_PATH))


        # First let's load meta graph and restore weights
        # self.saver = tf.train.import_meta_graph(cfg.CKPT_FILE + '-2385' + '.meta')
        # self.saver.restore(sess, cfg.CKPT_FILE + '-2385')

        # self.graph = tf.get_default_graph()
        # self.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        # print([n.name for n in tf.get_default_graph().as_graph_def().node])


        # self.model.X = self.graph.get_tensor_by_name('Vggnet/initialize_scope/X_data:0')
        # self.model.Y = self.graph.get_tensor_by_name('Vggnet/initialize_scope/Y_data:0')
        # self.model.training = self.graph.get_tensor_by_name('Vggnet/initialize_scope/training:0')


    def classify(self):
        print('Test Started!')




        validation_accuracy = []

        for idx in range(0, len(cfg.TRAIN_FILE_LIST), 2):
            # load train / validation dataset
            train_total_x, train_total_y, train_total_size = read_data(cfg.TRAIN_FILE_LIST[idx],
                                                                       cfg.TRAIN_FILE_LIST[idx + 1])
            # train 0.9 / validation 0.1
            train_x, train_y = train_total_x[0:int(0.9 * train_total_size)], train_total_y[
                                                                             0:int(0.9 * train_total_size)]
            validation_x, validation_y = train_total_x[int(0.9 * train_total_size):], train_total_y[
                                                                                      int(0.9 * train_total_size):]

            # validation
            for start_idx in range(0, len(validation_x), cfg.BATCH_SIZE):
                validation_x_batch = validation_x[start_idx:start_idx + cfg.BATCH_SIZE]
                validation_y_batch = validation_y[start_idx:start_idx + cfg.BATCH_SIZE]

                feed_dict = {self.model.X: validation_x_batch, self.model.Y: validation_y_batch,
                             self.model.training: False}

                validation_acc = self.sess.run([self.model.accuracy], feed_dict=feed_dict)
                validation_accuracy.append(validation_acc)

        print(validation_accuracy)
        print(np.mean(np.array(validation_accuracy)))

        test_accuracy = []

        for idx in range(0, len(cfg.TEST_FILE_LIST), 2):
            test_x, test_y, test_total_size = read_data(cfg.TEST_FILE_LIST[idx], cfg.TEST_FILE_LIST[idx + 1])

            for start_idx in range(0, test_total_size, cfg.BATCH_SIZE):
                test_x_batch = test_x[start_idx:start_idx + cfg.BATCH_SIZE]
                test_y_batch = test_y[start_idx:start_idx + cfg.BATCH_SIZE]

                feed_dict = {self.model.X: test_x_batch, self.model.Y: test_y_batch, self.model.training: False}

                acc = self.sess.run([self.model.accuracy], feed_dict=feed_dict)
                test_accuracy.append(acc)
        print(test_accuracy)
        print('Test Accuracy :', np.mean(np.array(test_accuracy)))
        print('Test Finished!')
        print('global_step :', self.sess.run(self.model.global_step))


def main():
    model = Vggnet(name='Vggnet', label_cnt=cfg.LABEL_CNT)
    classifier = Classifier(model)
    classifier.classify()

if __name__ == '__main__':
    main()