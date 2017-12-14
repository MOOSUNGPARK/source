import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import ImageGrab
from drawnow import drawnow
from Portpolio.Dog_Cat.Model_ensemble.ensemble3 import Model, ensemble_accuracy

epochs = 50
batch_size = 100
num_models = 5
label_cnt = 2
value_num = 3

mon_epoch_list = []
mon_value_list = [[] for _ in range(value_num)]
mon_color_list = ['blue', 'yellow', 'red', 'cyan', 'magenta', 'green', 'black']
mon_label_list = ['avg_loss', 'avg_train_acc', 'avg_val_acc']

train_file_list = ['c:\\python\\source\\Portpolio\\Dog_Cat\\data\\train_data_{}.csv'.format(i) for i in range(1,21)]
test_file_list = ['c:\\python\\source\\Portpolio\\Dog_Cat\\data\\test_data_{}.csv'.format(i) for i in range(1,11)]

def data_setting(data):
    total_size = len(data)
    x = (np.array(data[:,0:-1]) / 255).tolist()
    targets = data[:,-1].astype(np.int32)
    y = np.zeros((total_size,label_cnt))
    y[np.arange(total_size), targets] = 1
    return x,y,total_size

def read_data(*filename):
    temp = []
    for file in filename:
        temp.append(np.loadtxt(file, delimiter=','))
    data = np.concatenate(temp, axis=0)
    np.random.shuffle(data)
    return data_setting(data)

def image_screenshot():
    im = ImageGrab.grab()
    im.show()

def monitor_train_cost():
    for cost, color, label in zip(mon_value_list, mon_color_list[0:len(mon_label_list)], mon_label_list):
        plt.plot(mon_epoch_list, cost, c=color, lw=2, ls='--', marker='o', label=label)
    plt.title('Ensemble on Dog_cat')
    plt.legend(loc=1 ,prop={'size' : 6})
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.grid(True)

def get_model_params():
    gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    return {gvar.op.name: value for gvar, value in zip(gvars, tf.get_default_session().run(gvars))}

def restore_model_params(model_params):
    gvar_names = list(model_params.keys())
    assign_ops = {gvar_name: tf.get_default_graph().get_operation_by_name(gvar_name + '/Assign') for gvar_name in gvar_names}
    init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
    feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
    tf.get_default_session().run(assign_ops, feed_dict=feed_dict)


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7

with tf.Session(config=config) as sess:
    stime = time.time()
    models = []
    for m in range(num_models):
        models.append(Model(sess, 'model' + str(m+1), label_cnt))

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    best_loss_val = np.infty
    check_since_last_progress = 0
    max_checks_without_progress = int(epochs * 0.2)
    best_model_params = None

    print('Learning Started!')

    '''Data Loading'''

    for epoch in range(epochs):
        epoch_stime = time.time()
        train_accuracy = [[] for _ in range(num_models)]
        validation_accuracy = [[] for _ in range(num_models)]
        validation_loss_list = np.zeros(num_models)

        if epoch+1 in (int(0.5 * epochs), int(0.75 * epochs)):
            for idx, m in enumerate(models):
                m.learning_rate /= 10

        for idx in range(0, len(train_file_list), 2):
            train_total_x, train_total_y, train_total_size = read_data(train_file_list[idx], train_file_list[idx+1])
            train_x, train_y = train_total_x[0:int(0.9 * train_total_size)], train_total_y[0:int(0.9 * train_total_size)]
            validation_x, validation_y = train_total_x[int(0.9 * train_total_size):], train_total_y[int(0.9 * train_total_size):]

            '''train part'''
            for start_idx in range(0, len(train_x), batch_size):
                train_x_batch, train_y_batch = train_x[start_idx:start_idx+batch_size], train_y[start_idx:start_idx+batch_size]

                for i, m in enumerate(models):
                    a, _ = m.train(train_x_batch, train_y_batch)
                    train_accuracy[i].append(a)

            '''validation part'''
            for start_idx in range(0, len(validation_x), batch_size):
                validation_x_batch, validation_y_batch = validation_x[start_idx:start_idx+batch_size], \
                                                         validation_y[start_idx:start_idx+batch_size]
                for i, m in enumerate(models):
                    l, a = m.validation(validation_x_batch, validation_y_batch)
                    validation_loss_list[i] += l / batch_size
                    validation_accuracy[i].append(a)

        '''early stopping condition check'''
        if np.mean(validation_loss_list) < best_loss_val:
            best_loss_val = np.mean(validation_loss_list)
            check_since_last_progress = 0
            best_model_params = get_model_params()
            saver.save(sess, 'log/ensemble_dog_cat.ckpt')
        else:
            check_since_last_progress += 1

        mon_epoch_list.append(epoch + 1)
        mon_value_list[0].append(np.mean(validation_loss_list))
        mon_value_list[1].append(np.mean(np.array(train_accuracy)) * 100)
        mon_value_list[2].append(np.mean(np.array(validation_accuracy)) * 100)

        epoch_etime = time.time()
        print('epoch :', epoch + 1, ', time :', round(epoch_etime-epoch_stime, 6),
              '\n\t\tloss :', validation_loss_list,
              '\n\t\ttrain_accuracy :', [np.mean(np.array(train_accuracy[i])) * 100 for i in range(num_models)],
              '\n\t\tvalidation_accuracy :', [np.mean(np.array(validation_accuracy[i])) * 100 for i in range(num_models)])
        drawnow(monitor_train_cost)

        if check_since_last_progress > max_checks_without_progress:
            print('Early stopping!')
            break

    print('Learning Finished')

    etime = time.time()
    print('consumption time : ', round(etime-stime, 6))


    '''test part'''
    print('\nTesting Started!')

    if best_model_params:
        restore_model_params(best_model_params)

    total_accuracy_list = []
    model_accuracy_list = [0. for _ in range(num_models)]
    cnt = 0

    for idx in range(0, len(test_file_list), 2):
        test_x, test_y, test_total_size = read_data(test_file_list[idx], test_file_list[idx+1])

        for start_idx in range(0, test_total_size, batch_size):
            test_x_batch, test_y_batch = test_x[start_idx:start_idx+batch_size], test_y[start_idx:start_idx+batch_size]
            predict = np.zeros((len(test_x_batch), label_cnt))

            for i, m in enumerate(models):
                model_accuracy_list[i] += m.get_accuracy(test_x_batch, test_y_batch)
                predict += m.predict(test_x_batch)

            total_accuracy_list.append(ensemble_accuracy(predict, test_y_batch))
            cnt += 1

    for idx in range(num_models):
        print('Model {} Test Accuracy :'.format(idx), model_accuracy_list[idx] / cnt)
    print('Total Test Accuracy :', np.mean(np.array(total_accuracy_list)))
    print('Test Finished!')