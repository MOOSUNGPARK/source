import numpy as np


def save_file(data):
    # train_cnt = int(len(data) * 0.8)
    np.savetxt('data\\test_data.csv', data, delimiter=',', fmt='%d')
    # np.savetxt('data\\test_data.csv', data[train_cnt:], delimiter=',', fmt='%.1f')


def make_file(*filename):
    temp = []
    for file in filename:
        temp.append(np.loadtxt(file, delimiter=','))
    data = np.concatenate(temp, axis=0)
    np.random.shuffle(data)
    return save_file(data)


if __name__ == '__main__':
    make_file('C:\\python\\source\\Portpolio\\Dog_Cat\\data\\test_data_1.csv',
              'C:\\python\\source\\Portpolio\\Dog_Cat\\data\\test_data_2.csv',
              'C:\\python\\source\\Portpolio\\Dog_Cat\\data\\test_data_3.csv',
              'C:\\python\\source\\Portpolio\\Dog_Cat\\data\\test_data_4.csv',
              'C:\\python\\source\\Portpolio\\Dog_Cat\\data\\test_data_5.csv',
              'C:\\python\\source\\Portpolio\\Dog_Cat\\data\\test_data_6.csv',
              'C:\\python\\source\\Portpolio\\Dog_Cat\\data\\test_data_7.csv',
              'C:\\python\\source\\Portpolio\\Dog_Cat\\data\\test_data_8.csv',
              'C:\\python\\source\\Portpolio\\Dog_Cat\\data\\test_data_9.csv',
              'C:\\python\\source\\Portpolio\\Dog_Cat\\data\\test_data_10.csv')



