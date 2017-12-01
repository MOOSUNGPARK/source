from Portpolio.Face_recognition.cnn_model import Model
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
from drawnow import drawnow
from PIL import ImageGrab

batch_size = 100
num_models = 5

train_file_list = ['data/train_data_' + str(i) + '.csv' for i in range(1, 21)]
test_file_list = ['data/test_data_' + str(i) + '.csv' for i in range(1, 11)]

def data_setting(data):
    x = (np.array(data[:,0:-1]) / 255).tolist()
    y  = [[1,0] if y_ == 0 else [0,1] for y_ in data[:, [-1]]]
    return x,y

def read_data(*filename):
    data1 = np.loadtxt(filename[0], delimiter=',')
    data2 = np.loadtxt(filename[1], delimiter=',')
    data = np.append(data1, data2, axis=0)
    np.random.shuffle(data)
    return data_setting(data)

def image_screenshot():
    im = ImageGrab.grab()
    im.show()

mon_epoch_list = []
mon_cost_list = [[] for m in range(num_models)]
mon_color_list = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']
mon_label_list = ['model' + str(m+1) for m in range(num_models)]

def monitor_train_cost():
    for cost, color, label in zip(mon_cost_list, mon_color_list[0:len(mon_label_list)], mon_label_list):
        plt.plot(mon_epoch_list, cost, c=color, lw=2, ls='--', marker='o', label=label)
    plt.title('Epoch per Cost Graph')
    plt.legend(loc=1)
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.grid(True)





