import numpy as np


def save_file(data):
    train_cnt = int(len(data) * 0.8)
    np.savetxt('data\\train_data.csv', data[:train_cnt], delimiter=',', fmt='%.1f')
    np.savetxt('data\\test_data.csv', data[train_cnt:], delimiter=',', fmt='%.1f')


def make_file(*filename):
    temp = []
    for file in filename:
        temp.append(np.loadtxt(file, delimiter=','))
    data = np.concatenate(temp, axis=0)
    np.random.shuffle(data)
    return save_file(data)


if __name__ == '__main__':
    make_file('C:\\python\\source\\Portpolio\\Fer2013\\data\\fer_data1.csv')

# def save_file(data):
#     train_cnt = int(len(data) * 0.8)
#     np.savetxt('data\\train_data.csv', data[:train_cnt], delimiter=',', fmt='%.1f')
#     np.savetxt('data\\test_data.csv', data[train_cnt:], delimiter=',', fmt='%.1f')
#
#
# def make_file(*filename):
#     temp = []
#     for file in filename:
#         temp.append(np.loadtxt(file, delimiter=' '))
#     data = np.concatenate(temp, axis=0)
#     # np.random.shuffle(data)
#     # return save_file(data)
#     np.savetxt('data\\image_after.csv', data, delimiter=',', fmt='%.1f')
#
# if __name__ == '__main__':
#     make_file('C:\\python\\data\\Fer2013\\image_before.csv')

