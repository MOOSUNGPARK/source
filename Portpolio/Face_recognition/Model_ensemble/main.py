import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import ImageGrab
from drawnow import drawnow

from Portpolio.Face_recognition.Model_ensemble.cnn_model1 import Model

batch_size = 100
num_models = 5
label_cnt = 4
epochs = 20

train_file = 'c:\\python\\source\\Portpolio\\Face_recognition\\data\\train_data.csv'
test_file = 'c:\\python\\source\\Portpolio\\Face_recognition\\data\\test_data.csv'

def data_setting(data):
    total_size = len(data)
    x = (np.array(data[:,0:-1]) / 255).tolist()
    targets = data[:,-1].astype(np.int32)
    y = np.zeros((total_size, label_cnt))
    y[np.arange(total_size), targets] = 1
    return x,y,total_size

def read_data(filename):
    data = np.loadtxt(filename, delimiter=',')
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


with tf.Session() as sess:
    stime = time.time()

    models = []
    for m in range(num_models):
        models.append(Model(sess, 'model' + str(m+1), label_cnt))

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    print('Learning Started')

    early_stopping_list = []
    last_epoch = -1
    epoch = 0
    ep_check = 0

    while True:
        sstime = time.time()
        avg_cost_list = np.zeros(len(models))

        total_x, total_y, train_total_size  = read_data(train_file)

        for start_idx in range(0, train_total_size, batch_size):
            train_x_batch, train_y_batch = total_x[start_idx:start_idx+batch_size], total_y[start_idx:start_idx+batch_size]

            for i, m in enumerate(models):
                c, _ = m.train(train_x_batch, train_y_batch)
                avg_cost_list[i] += c / batch_size

        mon_epoch_list.append(epoch+1)
        for idx, cost in enumerate(avg_cost_list):
            mon_cost_list[idx].append(cost)
        drawnow(monitor_train_cost)

        saver.save(sess, 'log/epoch_' + str(epoch+1) + '.ckpt')
        early_stopping_list.append(avg_cost_list)
        diff = 0

        if len(early_stopping_list) >= 2:
            temp = np.array(early_stopping_list)
            last_epoch = epoch
            diff = np.sum(temp[0] < temp[1])

            if diff > 2:
                print('Epoch: ', '%04d' % (epoch+1), 'cost =', avg_cost_list, ' - ', diff)
                print('early stopping - epoch({})'.format(epoch+1))
                ep_check += 1
            early_stopping_list.pop(0)
        epoch += 1
        if epoch == epochs:
            break
        eetime = time.time()
        print('Epoch: ', '%04d' % (epoch), 'cost = ', avg_cost_list, ' - ', diff, ', epoch{} time'.format(epoch),
              round(eetime - sstime, 2), ', ep_check', ep_check)
    print('Learning Finished!')


    image_screenshot()
    etime = time.time()
    print('consumption time : ', round(etime - stime, 6))

tf.reset_default_graph()


########################################################
with tf.Session() as sess:
    models = []
    num_models = 5

    for idx in range(num_models):
        models.append(Model(sess, 'model' + str(idx+1), label_cnt))

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, 'log/epoch_' + str(last_epoch) + '.ckpt')

    print('Testing Started')

    ensemble_accuracy = 0.
    model_accuracy = [0., 0., 0., 0., 0.]
    cnt = 0

    total_x, total_y, test_total_size = read_data(test_file)

    for start_idx in range(0, test_total_size, batch_size):
        test_x_batch, test_y_batch = total_x[start_idx:start_idx+batch_size], total_y[start_idx:start_idx+batch_size]
        test_size = len(test_y_batch)
        predictions = np.zeros(test_size * 10).reshape(test_size, 10)

        model_result = np.zeros(test_size*2, dtype=np.int).reshape(test_size, 2)
        model_result[:,0] = range(0,test_size)

        for i, m in enumerate(models):
            model_accuracy[i] += m.get_accuracy(test_x_batch, test_y_batch)
            p = m.predict(test_x_batch)
            model_result[:,1] = np.argmax(p,1)

            for result in model_result:
                predictions[result[0], result[1]] += 1

        ensemble_correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(test_y_batch, 1))
        ensemble_accuracy += tf.reduce_mean(tf.cast(ensemble_correct_prediction, tf.float32))
        cnt += 1

    for i in range(len(model_accuracy)):
        print('Model ' + str(i) + ' : ', model_accuracy[i] / cnt)
    print('Ensemble Accuracy : ', sess.run(ensemble_accuracy) / cnt)
    print('Testing Finished!')


