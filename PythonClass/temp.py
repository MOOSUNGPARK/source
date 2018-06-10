import os
import numpy as np
#

# def rgb(pred_img, color):
#     rgb_dic = {'green': 0, 'blue': 1, 'red': 2}
#     rgb_list = [np.zeros([1, 2, 2, 1]) for _ in range(3)]
#     rgb_list[rgb_dic[color]] = pred_img
#
#     return rgb_list

# rgb_dic = {'green': 0, 'blue': 1, 'red': 2}
# print(rgb_dic['red'])
# rgb_list = [ np.zeros([1, 2, 2, 1]),
#              np.zeros([1, 2, 2, 1]),
#              np.zeros([1, 2, 2, 1])]
# rgb_list2 = [ np.zeros([1, 2, 2, 1]) for _ in range(3)]
# print(np.shape(rgb_list))
# print(np.shape(rgb_list2))



# a = [[[1,2,3,4], [1,2,3,4]],[[1,2,3,4], [1,2,3,4]]]
# print(np.shape(a))
# print(len(np.shape(a)))

# a, b, c = [], [], []
# print(a)
# print(b)
# print(c)

a = 'label2'
b = 'label'

# if 'label' in a :
#     print('a')
#
# if 'label' in b:
#     print('b')

dir_name = ['merged', 'pred', 'label', 'compare']
path_list = [('{0}imgs{0}{1}{0}' + name).format(1, 2) for name in dir_name]
print(path_list)