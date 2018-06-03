import numpy as np
from PIL import ImageGrab
import Portpolio.Cifar10_classify.Model_shufflenet.config as cfg


# http://leechanho.tistory.com/16
# class CSV_reader():
#     def __init__(self, sess):
#         self.sess = sess
#         self.batch_size = cfg.BATCH_SIZE
#
#
#     def read_batch_data(self, file_list):
#         filename_queue = tf.train.string_input_producer(file_list, shuffle=False, name='filename_queue')
#         reader = tf.TextLineReader()
#         _, value = reader.read(filename_queue)
#         record_defaults = [[0.] for _ in range(126 * 126)] + [[0]] # X: 126 * 126 -> tf.float  / Y: 1 -> tf.int
#         xy = tf.decode_csv(value, record_defaults=record_defaults)
#         x_batch, y_batch = tf.train.batch([xy[0:-1], xy[-1:]], batch_size=self.batch_size)
#
#         return x_batch, y_batch

def data_setting(data):
    total_size = len(data)
    x = (np.array(data[:,0:-1]) / 255).tolist()
    targets = data[:,-1].astype(np.int32)
    y = np.zeros((total_size, cfg.LABEL_CNT))
    y[np.arange(total_size), targets] = 1
    return x,y, total_size

def read_data(*filename):
    temp = []
    for file in filename:
        temp.append(np.loadtxt(file, delimiter=','))
    data = np.concatenate(temp, axis=0)
    # np.random.shuffle(data)
    return data_setting(data)

# def monitor_train_cost(mon_epoch_list, mon_value_list, mon_color_list, mon_label_list):
#     for cost, color, label in zip(mon_value_list, mon_color_list[0:len(mon_label_list)], mon_label_list):
#         plt.plot(mon_epoch_list, cost, c=color, lw=2, ls='--', marker='o', label=label)
#     plt.title('Mobilenet on Dog_Cat')
#     plt.legend(loc=1)
#     plt.xlabel('Epoch')
#     plt.ylabel('Value')
#     plt.grid(True)

def image_screenshot():
    im = ImageGrab.grab()
    im.show()

