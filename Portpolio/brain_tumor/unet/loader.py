"""
데이터 로더입니다.

이미지 사이즈와 백그라운드 라벨 생성 여부를 전달받고 각각의 메소드를 통해 데이터의 절대경로를 만들고 이미지를 불러와서 배치데이터를 만들어줍니다.

numpy와 cv2를 사용하여 구성되어있습니다.


"""

import re
import cv2
import os
import numpy as np


# class DataLoader:
#     def __init__(self, img_size, background_label=None):
#         self.img_size = img_size
#         self.background_label = background_label
#
#     # 파일명에 숫자가 있을 경우 숫자 순으로 정렬하는 함수입니다. _number_key()를 사용하면 됩니다.
#     def _try_int(self, ss):
#         try:
#             return int(ss)
#         except:
#             return ss
#
#     def _number_key(self, s):
#         return [self._try_int(ss) for ss in re.split('([0-9]+)', s)]
#
#     def _sort_by_number(self, files):
#         files.sort(key=self._number_key)
#         return files
#
#
#     # 데이터 경로 로더입니다.
#     def data_list_load(self, path, mode):
#
#         if mode == 'train':
#             # 데이터셋 경로를 담아 둘 빈 리스트 생성
#             image_list = []
#             label_list = []
#
#             # 입력된 모든 경로에 대해서 이미지 데이터 경로를 절대경로로 만든 다음 위에서 생성한 리스트에 저장하고 반환
#             for data_path in path:
#                 for root, dirs, files in os.walk(data_path):
#                     for dir in dirs:
#                         dir_path = os.path.join(root, dir)
#                         # 윈도우에서는 '/x' 대신 '\\x'로 진행해야 합니다.
#                         if '/x' in dir_path:
#                             if len(os.listdir(dir_path)) != 0:
#                                 # 각 파일의 절대경로를 만드는 리스트 컴프리헨션입니다.
#                                 x_path_list = [os.path.join(dir_path, file) for file in os.listdir(dir_path)]
#                                 y_path_list = [os.path.join(dir_path, file) for file in os.listdir(dir_path)]
#                                 y_path_list = [path.replace('/x/', '/y/') for path in y_path_list]
#
#                                 # 위 리스트를 파일명 숫자 순서로 정렬합니다.
#                                 images_files = self._sort_by_number(x_path_list)
#                                 labels_files = self._sort_by_number(y_path_list)
#
#                                 for image in images_files:
#                                     image_list.append(image)
#                                     # print('xdata:', image)
#
#                                 for label in labels_files:
#                                     label_list.append(label)
#                                     # print('ydata:', label)
#
#             return image_list, label_list, len(image_list)
#
#         # 테스트 모드의 경우 라벨데이터를 불러올 필요가 없으므로 따로 생성합니다.
#         elif mode == 'test':
#             image_list = []
#             for data_path in path:
#                 for root, dirs, files in os.walk(data_path):
#                     for dir in dirs:
#                         dir_path = os.path.join(root, dir)
#                         if '/x' in dir_path:
#                             if len(os.listdir(dir_path)) != 0:
#                                 x_path_list = [os.path.join(dir_path, file) for file in os.listdir(dir_path)]
#
#                                 images_files = self._sort_by_number(x_path_list)
#
#                                 for image in images_files:
#                                     image_list.append(image)
#
#             return image_list, len(image_list)
#
#     # 배치 iteration을 수행하는 메소드입니다.
#     def next_batch(self, data, idx, batch_size, mode, label=None):
#         #                             'validation'
#         if mode == 'train' or mode == 'validation':
#             data_list = np.array(data)
#             label_list = np.array(label)
#
#             batch_data_list = data_list[idx * batch_size:idx * batch_size + batch_size]
#             batch_label_list = label_list[idx * batch_size:idx * batch_size + batch_size]
#
#             index = np.arange(len(batch_data_list))
#             np.random.shuffle(index)
#             batch = batch_data_list[index]
#             label = batch_label_list[index]
#
#             return batch, label
#
#         elif mode == 'validation':
#             data_list = np.array(data)
#             label_list = np.array(label)
#
#             batch_data_list = data_list[idx * batch_size:idx * batch_size + batch_size]
#             batch_label_list = label_list[idx * batch_size:idx * batch_size + batch_size]
#
#             index = np.arange(len(batch_data_list))
#             batch = batch_data_list[index]
#             label = batch_label_list[index]
#
#             return batch, label
#
#         elif mode == 'test':
#             data_list = np.array(data)
#             batch_data_list = data_list[idx * batch_size:idx * batch_size + batch_size]
#
#             index = np.arange(len(batch_data_list))
#             batch = batch_data_list[index]
#
#             return batch
#
#     # 데이터를 랜덤 셔플하는 메소드입니다.
#     def data_shuffle(self, data, label):
#         data = np.array(data)
#         label = np.array(label)
#
#         index = np.arange(len(data))
#         np.random.shuffle(index)
#
#         data = data[index]
#         label = label[index]
#
#         return data, label
#
#     # 데이터를 학습데이터와 평가데이터셋으로 비율에 맞춰서 분리하는 메소드입니다.
#     def data_split(self, data, label, val_size):
#         data_count = len(data)
#         if round(data_count* (val_size / 100)) == 0:
#             val_data_cnt = 1
#         else:
#             val_data_cnt = round(data_count * (val_size / 100))
#
#         trainX = data[:-val_data_cnt]
#         trainY = label[:-val_data_cnt]
#         valX = data[-val_data_cnt:]
#         valY = label[-val_data_cnt:]
#
#         return trainX, trainY, valX, valY
#
#     # Data augmentation 관련 메소드 들입니다. _idle은 원본이미지 그대로 나가며 flip은 좌우나 상하 반전, rotate는 회전을 시킵니다.
#     def _idle(self, img):
#         return img
#
#     def flipImage(self, img, option_number):
#         # 이미지 반전,  0:상하, 1 : 좌우로 반전합니다.
#         img = cv2.flip(img, option_number)
#         return img
#
#     def rotateImage(self, image, angle):
#         # 이미지 회전은 cv2.getRotationMatrix2D로 회전시킬 매트릭스를 생성한 다음 cv2.warpAffine으로 회전합니다.
#         image_center = tuple(np.array(image.shape[1::-1]) / 2)
#         rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
#         result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
#         return result
#
#     # 옵션 난수를 받아서 augmentation을 실행합니다.
#     def _data_flip(self, img, augment_number):
#         # case = {
#         #     0: self._idle(img),
#         #     1: self.flipImage(img, 1),
#         #     2: self.flipImage(img, 0),
#         #
#         # }
#         # return case[augment_number]
#         if augment_number == 1:
#             return self.flipImage(img, 1)
#         elif augment_number == 2:
#             return self.flipImage(img, 0)
#         else:
#             return self._idle(img)
#
#     def _data_rotation(self, img, augment_number):
#         # case = {
#         #     0: self._idle(img),
#         #     1: self.rotateImage(img, 10),
#         #     2: self.rotateImage(img, -10),
#         #     3: self.rotateImage(img, 20),
#         #     4: self.rotateImage(img, -20)
#         # }
#         # return case[augment_number]
#         if augment_number == 1:
#             return self.rotateImage(img, 10)
#         elif augment_number == 2:
#             return self.rotateImage(img, -10)
#         elif augment_number == 3:
#             return self.rotateImage(img, 20)
#         elif augment_number == 4:
#             return self.rotateImage(img, -20)
#         else:
#             return self._idle(img)
#
#     def aug_percentage(self, percentage):
#         flip_percentage_n = (2*percentage)+2
#         rot_percentage_n = (4*percentage)+4
#         return flip_percentage_n, rot_percentage_n
#
#     # 이미지파일을 읽어오는 메소드입니다.
#     def read_data(self, mode, x_list, y_list=None, augmentation=None, augmentation_pergentage=None, channel_mode='NHWC'):
#         # train
#         if mode == 'train':
#             # 어규먼테이션 실행
#             if augmentation is True:
#                 if len(x_list) != len(y_list):
#                     raise AttributeError('The amounts of X and Y data are not equal.')
#
#                 else:
#                     # x
#                     if type(x_list) != str:
#                         x_list = x_list
#                     elif type(x_list) == str:
#                         x_list = [x_list]
#
#                     # y
#                     if type(y_list) != str:
#                         y_list = y_list
#                     elif type(y_list) == str:
#                         y_list = [y_list]
#
#                     x_data = []
#                     y_data = []
#
#                     for i in range(len(x_list)):
#                         # augmentation을 랜덤하게 진행하기 위해 옵션값으로 사용할 난수를 생성합니다.
#                         flip_per, rot_per = self.aug_percentage(augmentation_pergentage)
#                         random_number_for_flip = int(np.random.randint(0, flip_per, size=1)[0])
#                         random_number_for_rotate = int(np.random.randint(0, rot_per, size=1)[0])
#
#                         # 학습데이터
#                         # 절대경로를 받아서 이미지를 Grayscale로 읽어오고 augmentation을 진행한 다음 동일 사이즈로 리사이징 합니다.
#                         x_img = cv2.imread(x_list[i], cv2.IMREAD_GRAYSCALE)
#                         x_img = self._data_rotation(self._data_flip(x_img, random_number_for_flip), random_number_for_rotate)
#
#                         # 입력받은 이미지 사이즈로 일괄 리사이징을 진행합니다. 리사이징을 할 때 보정효과를 넣어주는 인터폴레이션 옵션은 여러종류가 있으니 cv2 홈페이지를 참고하세요.
#                         x_img = cv2.resize(x_img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
#
#                         x_data.append(x_img)
#
#                         # 라벨데이터
#                         y_img = cv2.imread(y_list[i], cv2.IMREAD_GRAYSCALE)
#                         y_img = self._data_rotation(self._data_flip(y_img, random_number_for_flip), random_number_for_rotate)
#                         y_img = cv2.resize(y_img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
#
#                         # 리사이징 과정에서 픽셀변화로 바이너리화된 라벨데이터가 픽셀값 변동이 생기므로 다시 thresholding 해줍니다.
#                         y_img1 = cv2.threshold(y_img, 124, 255, cv2.THRESH_BINARY)[1]
#                         y_img1 = y_img1.reshape([self.img_size, self.img_size, 1])
#
#                         # 백그라운드 라벨링이 True면 cv2.THRESH_BINARY_INV로 백그라운드에 라벨링을 수행하여 2채널로 라벨데이터를 리턴합니다.
#                         # 백그라운드 라벨링은 학습과정에서 백그라운드를 알려주고 필터에 노이즈를 주어 학습성능을 향상시키는 효과가 있습니다.
#                         if self.background_label is True:
#                             y_img2 = cv2.threshold(y_img, 124, 255, cv2.THRESH_BINARY_INV)[1]
#                             y_img2 = y_img2.reshape([self.img_size, self.img_size, 1])
#                             y_img = np.concatenate((y_img1, y_img2), axis=2)
#                             y_data.append(y_img)
#                         else:
#                             y_data.append(y_img)
#
#                     if self.background_label is True:
#                         if channel_mode == 'NHWC':
#                             return np.array(x_data).reshape([-1, self.img_size, self.img_size, 1]), np.array(y_data).reshape([-1, self.img_size, self.img_size, 2])
#                         elif channel_mode == 'NCHW':
#                             return np.array(x_data).reshape([-1, 1, self.img_size, self.img_size]), np.array(y_data).reshape([-1, 2, self.img_size, self.img_size])
#                     else:
#                         if channel_mode == 'NHWC':
#                             return np.array(x_data).reshape([-1, self.img_size, self.img_size, 1]), np.array(y_data).reshape([-1, self.img_size, self.img_size, 1])
#                         elif channel_mode == 'NCHW':
#                             return np.array(x_data).reshape([-1, 1, self.img_size, self.img_size]), np.array(y_data).reshape([-1, 1, self.img_size, self.img_size])
#
#             # 어규먼테이션 실행 안함
#             else:
#                 if len(x_list) != len(y_list):
#                     raise AttributeError('The amounts of X and Y data are not equal.')
#
#                 else:
#                     # x
#                     if type(x_list) != str:
#                         x_list = x_list
#                     elif type(x_list) == str:
#                         x_list = [x_list]
#
#                     # y
#                     if type(y_list) != str:
#                         y_list = y_list
#                     elif type(y_list) == str:
#                         y_list = [y_list]
#
#                     x_data = []
#                     y_data = []
#
#                     for i in range(len(x_list)):
#                         x_img = cv2.imread(x_list[i], cv2.IMREAD_GRAYSCALE)
#
#                         x_img = cv2.resize(x_img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
#                         x_data.append(x_img)
#
#                         y_img = cv2.imread(y_list[i], cv2.IMREAD_GRAYSCALE)
#                         y_img = cv2.resize(y_img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
#
#                         y_img1 = cv2.threshold(y_img, 124, 255, cv2.THRESH_BINARY)[1]
#                         y_img1 = y_img1.reshape([self.img_size, self.img_size, 1])
#
#                         if self.background_label is True:
#                             y_img2 = cv2.threshold(y_img, 124, 255, cv2.THRESH_BINARY_INV)[1]
#                             y_img2 = y_img2.reshape([self.img_size, self.img_size, 1])
#                             y_img = np.concatenate((y_img1, y_img2), axis=2)
#                             y_data.append(y_img)
#                         else:
#                             y_data.append(y_img)
#
#                     if self.background_label is True:
#                         if channel_mode == 'NHWC':
#                             return np.array(x_data).reshape([-1, self.img_size, self.img_size, 1]), np.array(y_data).reshape([-1, self.img_size, self.img_size, 2])
#                         elif channel_mode == 'NCHW':
#                             return np.array(x_data).reshape([-1, 1, self.img_size, self.img_size]), np.array(y_data).reshape([-1, 2, self.img_size, self.img_size])
#                     else:
#                         if channel_mode == 'NHWC':
#                             return np.array(x_data).reshape([-1, self.img_size, self.img_size, 1]), np.array(y_data).reshape([-1, self.img_size, self.img_size, 1])
#                         elif channel_mode == 'NCHW':
#                             return np.array(x_data).reshape([-1, 1, self.img_size, self.img_size]), np.array(y_data).reshape([-1, 1, self.img_size, self.img_size])
#
#         # validation
#         elif mode == 'validation':
#             if len(x_list) != len(y_list):
#                 raise AttributeError('The amounts of X and Y data are not equal.')
#
#             else:
#                 # x
#                 if type(x_list) != str:
#                     x_list = x_list
#                 elif type(x_list) == str:
#                     x_list = [x_list]
#
#                 # y
#                 if type(y_list) != str:
#                     y_list = y_list
#                 elif type(y_list) == str:
#                     y_list = [y_list]
#
#                 x_data = []
#                 y_data = []
#
#                 for i in range(len(x_list)):
#                     x_img = cv2.imread(x_list[i], cv2.IMREAD_GRAYSCALE)
#                     x_img = cv2.resize(x_img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
#                     x_data.append(x_img)
#
#                     y_img = cv2.imread(y_list[i], cv2.IMREAD_GRAYSCALE)
#                     y_img = cv2.resize(y_img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
#
#                     y_img1 = cv2.threshold(y_img, 124, 255, cv2.THRESH_BINARY)[1]
#                     y_img1 = y_img1.reshape([self.img_size, self.img_size, 1])
#
#                     if self.background_label is True:
#                         y_img2 = cv2.threshold(y_img, 124, 255, cv2.THRESH_BINARY_INV)[1]
#                         y_img2 = y_img2.reshape([self.img_size, self.img_size, 1])
#                         y_img = np.concatenate((y_img1, y_img2), axis=2)
#                         y_data.append(y_img)
#                     else:
#                         y_data.append(y_img)
#
#                 if self.background_label is True:
#                     if channel_mode == 'NHWC':
#                         return np.array(x_data).reshape([-1, self.img_size, self.img_size, 1]), np.array(y_data).reshape([-1, self.img_size, self.img_size, 2])
#                     elif channel_mode == 'NCHW':
#                         return np.array(x_data).reshape([-1, 1, self.img_size, self.img_size]), np.array(y_data).reshape([-1, 2, self.img_size, self.img_size])
#                 else:
#                     if channel_mode == 'NHWC':
#                         return np.array(x_data).reshape([-1, self.img_size, self.img_size, 1]), np.array(y_data).reshape([-1, self.img_size, self.img_size, 1])
#                     elif channel_mode == 'NCHW':
#                         return np.array(x_data).reshape([-1, 1, self.img_size, self.img_size]), np.array(y_data).reshape([-1, 1, self.img_size, self.img_size])
#
#         elif mode == 'test':
#             x_data = []
#
#             for i in range(len(x_list)):
#                 x_img = cv2.imread(x_list[i], cv2.IMREAD_GRAYSCALE)
#                 x_img = cv2.resize(x_img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
#                 x_data.append(x_img)
#
#             if channel_mode == 'NHWC':
#                 return np.array(x_data).reshape([-1, self.img_size, self.img_size, 1])
#             elif channel_mode == 'NCHW':
#                 return np.array(x_data).reshape([-1, 1, self.img_size, self.img_size])
#
#
#
#
#
# ############################################################
# #                    OLD CODE BACKUP                       #
# ############################################################
#
# # def read_image_grey_resized(self, data_list):
# #     if type(data_list) != str:
# #         data_list = data_list
# #     elif type(data_list) == str:
# #         data_list = [data_list]
# #
# #     data = []
# #     for file in data_list:
# #         img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
# #         img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
# #
# #         data.append(img)
# #
# #     return np.array(data).reshape([-1, self.img_size, self.img_size, 1])
# #
# # def read_label_grey_resized(self, data_list):
# #     if type(data_list) != str:
# #         data_list = data_list
# #     elif type(data_list) == str:
# #         data_list = [data_list]
# #
# #     data = []
# #     for file in data_list:
# #         img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
# #         img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
# #         img1 = cv2.threshold(img, 50, 1, cv2.THRESH_BINARY)[1]
# #         img2 = cv2.threshold(img, 50, 1, cv2.THRESH_BINARY_INV)[1]
# #         img1 = img1.reshape([self.img_size, self.img_size, 1])
# #         img2 = img2.reshape([self.img_size, self.img_size, 1])
# #         img = np.concatenate((img1, img2), axis=2)
# #         # print(img)
# #         data.append(img)
# #
# #     return np.array(data).reshape([-1, self.img_size, self.img_size, 2])
#
# #
# #     if len(x_list) != len(y_list):
# #         raise AttributeError('The amounts of X and Y data are not equal.')
# #
# #     else:
# #         # x
# #         if type(x_list) != str:
# #             x_list = x_list
# #         elif type(x_list) == str:
# #             x_list = [x_list]
# #
# #         # y
# #         if type(y_list) != str:
# #             y_list = y_list
# #         elif type(y_list) == str:
# #             y_list = [y_list]
# #
# #         x_data = []
# #         y_data = []
# #
# #         # 학습모드인 경우
# #         if mode == 'train':
# #             for i in range(len(x_list)):
# #                 # augmentation을 랜덤하게 진행하기 위해 옵션값으로 사용할 난수를 생성합니다.
# #                 if augmentation is True:
# #                     flip_per, rot_per = self.aug_percentage(augmentation_pergentage)
# #                     random_number_for_flip = int(np.random.randint(0, flip_per, size=1)[0])
# #                     random_number_for_rotate = int(np.random.randint(0, rot_per, size=1)[0])
# #                 else:
# #                     random_number_for_flip = 0
# #                     random_number_for_rotate = 0
# #
# #                 # 학습데이터
# #                 # 절대경로를 받아서 이미지를 Grayscale로 읽어오고 augmentation을 진행한 다음 동일 사이즈로 리사이징 합니다.
# #                 x_img = cv2.imread(x_list[i], cv2.IMREAD_GRAYSCALE)
# #                 x_img = self._data_rotation(self._data_flip(x_img, random_number_for_flip), random_number_for_rotate)
# #
# #                 # 입력받은 이미지 사이즈로 일괄 리사이징을 진행합니다. 리사이징을 할 때 보정효과를 넣어주는 인터폴레이션 옵션은 여러종류가 있으니 cv2 홈페이지를 참고하세요.
# #                 x_img = cv2.resize(x_img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
# #
# #                 x_data.append(x_img)
# #
# #                 # 라벨데이터
# #                 y_img = cv2.imread(y_list[i], cv2.IMREAD_GRAYSCALE)
# #                 y_img = self._data_rotation(self._data_flip(y_img, random_number_for_flip), random_number_for_rotate)
# #                 y_img = cv2.resize(y_img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
# #
# #                 # 리사이징 과정에서 픽셀변화로 바이너리화된 라벨데이터가 픽셀값 변동이 생기므로 다시 thresholding 해줍니다.
# #                 y_img1 = cv2.threshold(y_img, 124, 255, cv2.THRESH_BINARY)[1]
# #                 y_img1 = y_img1.reshape([self.img_size, self.img_size, 1])
# #
# #                 # 백그라운드 라벨링이 True면 cv2.THRESH_BINARY_INV로 백그라운드에 라벨링을 수행하여 2채널로 라벨데이터를 리턴합니다.
# #                 # 백그라운드 라벨링은 학습과정에서 백그라운드를 알려주고 필터에 노이즈를 주어 학습성능을 향상시키는 효과가 있습니다.
# #                 if self.background_label is True:
# #                     y_img2 = cv2.threshold(y_img, 124, 255, cv2.THRESH_BINARY_INV)[1]
# #                     y_img2 = y_img2.reshape([self.img_size, self.img_size, 1])
# #                     y_img = np.concatenate((y_img1, y_img2), axis=2)
# #                     y_data.append(y_img)
# #                 else:
# #                     y_data.append(y_img)
# #
# #             if self.background_label is True:
# #                 return np.array(x_data).reshape([-1, self.img_size, self.img_size, 1]), np.array(y_data).reshape([-1, self.img_size, self.img_size, 2])
# #             else:
# #                 return np.array(x_data).reshape([-1, self.img_size, self.img_size, 1]), np.array(y_data).reshape([-1, self.img_size, self.img_size, 1])
# #
# #         # 밸리데이션인 경우
# #         elif mode == 'validation':
# #             for i in range(len(x_list)):
# #                 x_img = cv2.imread(x_list[i], cv2.IMREAD_GRAYSCALE)
# #                 x_img = cv2.resize(x_img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
# #                 x_data.append(x_img)
# #
# #                 y_img = cv2.imread(y_list[i], cv2.IMREAD_GRAYSCALE)
# #                 y_img = cv2.resize(y_img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
# #                 y_img1 = cv2.threshold(y_img, 124, 255, cv2.THRESH_BINARY)[1]
# #                 y_img1 = y_img1.reshape([self.img_size, self.img_size, 1])
# #
# #                 if self.background_label is True:
# #                     y_img2 = cv2.threshold(y_img, 124, 255, cv2.THRESH_BINARY_INV)[1]
# #                     y_img2 = y_img2.reshape([self.img_size, self.img_size, 1])
# #                     y_img = np.concatenate((y_img1, y_img2), axis=2)
# #                     y_data.append(y_img)
# #                 else:
# #                     y_data.append(y_img)
# #
# #             if self.background_label is True:
# #                 return np.array(x_data).reshape([-1, self.img_size, self.img_size, 1]), np.array(y_data).reshape([-1, self.img_size, self.img_size, 2])
# #             else:
# #                 return np.array(x_data).reshape([-1, self.img_size, self.img_size, 1]), np.array(y_data).reshape([-1, self.img_size, self.img_size, 1])
# #
# #     # 테스트인 경우
# # elif mode == 'test':
# #     # x
# #     if type(x_list) != str:
# #         x_list = x_list
# #     elif type(x_list) == str:
# #         x_list = [x_list]
# #
# #     x_data = []
# #     for i in range(len(x_list)):
# #          x_img = cv2.imread(x_list[i], cv2.IMREAD_GRAYSCALE)
# #          x_img = cv2.resize(x_img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
# #          x_data.append(x_img)
# #
# #     return np.array(x_data).reshape([-1, self.img_size, self.img_size, 1])


import re
import cv2
import os
import numpy as np


class DataLoader:
    def __init__(self, img_size):
        self.img_size = img_size

    def _try_int(self, ss):
        try:
            return int(ss)
        except:
            return ss

    def _number_key(self, s):
        return [self._try_int(ss) for ss in re.split('([0-9]+)', s)]

    # 파일명 번호 순으로 정렬
    def _sort_by_number(self, files):
        files.sort(key=self._number_key)
        return files

    # 데이터 경로 로더
    def data_list_load(self, path, mode):
        if mode == 'train':
            # 데이터셋 경로를 담아 둘 빈 리스트 생성
            image_list = []
            label_list = []

            # 입력된 모든 경로에 대해서 이미지 데이터 경로를 절대경로로 만든 다음 위에서 생성한 리스트에 저장하고 반환
            for data_path in path:
                for root, dirs, files in os.walk(data_path):
                    for dir in dirs:
                        dir_path = os.path.join(root, dir)
                        # windows에서는 path가 안 읽힘 : \x나 그런 식으로 바꿔야 될듯함.
                        if '/x' in dir_path:
                            if len(os.listdir(dir_path)) != 0:

                                x_path_list = [os.path.join(dir_path, file) for file in os.listdir(dir_path)]

                                y_path_list = [os.path.join(dir_path, file) for file in os.listdir(dir_path)]
                                # y_path_list = [path.replace('/x/', '/x_filtered/') for path in y_path_list]
                                y_path_list = [path.replace('/x/', '/y/') for path in y_path_list]

                                images_files = self._sort_by_number(x_path_list)
                                labels_files = self._sort_by_number(y_path_list)

                                for image in images_files:
                                    image_list.append(image)
                                    # print('xdata:', image)

                                for label in labels_files:
                                    label_list.append(label)
                                    # print('ydata:', label)

            return image_list, label_list, len(image_list)

        elif mode == 'test':
            # 데이터셋 경로를 담아 둘 빈 리스트 생성
            image_list = []
            down_list = []

            # 입력된 모든 경로에 대해서 이미지 데이터 경로를 절대경로로 만든 다음 위에서 생성한 리스트에 저장하고 반환
            for data_path in path:
                for root, dirs, files in os.walk(data_path):
                    for dir in dirs:
                        dir_path = os.path.join(root, dir)
                        if '/x' in dir_path:
                            if len(os.listdir(dir_path)) != 0:
                                x_path_list = [os.path.join(dir_path, file) for file in os.listdir(dir_path)]

                                images_files = self._sort_by_number(x_path_list)

                                for image in images_files:
                                    image_list.append(image)

            return image_list, len(image_list)

    def next_batch(self, data_list, label, idx, batch_size):
        data_list = np.array(data_list)
        label = np.array(label)

        batch1 = data_list[idx * batch_size:idx * batch_size + batch_size]
        label2 = label[idx * batch_size:idx * batch_size + batch_size]

        index = np.arange(len(batch1))
        np.random.shuffle(index)
        batch1 = batch1[index]
        label2 = label2[index]

        return batch1, label2

    def next_batch_test(self, data_list, idx, batch_size):
        data_list = np.array(data_list)

        batch = data_list[idx * batch_size:idx * batch_size + batch_size]

        index = np.arange(len(batch))
        np.random.shuffle(index)
        batch = batch[index]

        return batch

    def data_shuffle(self, data, label):
        data = np.array(data)
        label = np.array(label)

        index = np.arange(len(data))
        np.random.shuffle(index)

        data = data[index]
        label = label[index]

        return data, label

    def data_split(self, data, label, val_size):
        data_count = len(data)
        if round(data_count * (val_size / 100)) == 0:
            val_data_cnt = 1
        else:
            val_data_cnt = round(data_count * (val_size / 100))

        trainX = data[:-val_data_cnt]
        trainY = label[:-val_data_cnt]
        valX = data[-val_data_cnt:]
        valY = label[-val_data_cnt:]

        return trainX, trainY, valX, valY

    # 1. INTER_NEAREST is the fastest method and creates blocky images by just
    # choosing 1 pixel to replace several pixels.
    # 2. INTER_AREA is a fast method that gets the average of several pixels, which
    # is good for shrinking an image but not so good for enlarging an image.
    # 3. INTER_LINEAR uses bilinear interpolation to resize the image by combining
    # several pixels nicely (the best choice in many situations).
    # 4. INTER_CUBIC uses bicubic interpolation to use more advanced resizing that
    # linear, so is slightly slower but looks slightly better sometimes and
    # slightly worse other times.

    def read_image_grey_resized(self, data_list):
        if type(data_list) != str:
            data_list = data_list
        elif type(data_list) == str:
            data_list = [data_list]

        data = []
        for file in data_list:
            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)

            data.append(img)

        return np.array(data).reshape([-1, self.img_size, self.img_size, 1])

    def read_label_grey_resized(self, data_list):
        if type(data_list) != str:
            data_list = data_list
        elif type(data_list) == str:
            data_list = [data_list]

        data = []
        for file in data_list:
            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
            img1 = cv2.threshold(img, 50, 1, cv2.THRESH_BINARY)[1]
            img2 = cv2.threshold(img, 50, 1, cv2.THRESH_BINARY_INV)[1]
            img1 = img1.reshape([self.img_size, self.img_size, 1])
            img2 = img2.reshape([self.img_size, self.img_size, 1])
            img = np.concatenate((img1, img2), axis=2)
            # print(img)
            data.append(img)

        return np.array(data).reshape([-1, self.img_size, self.img_size, 2])

    #
    # # Data augmentation 관련 메소드 들입니다. _idle은 원본이미지 그대로 나가며 flip은 좌우나 상하 반전, rotate는 회전을 시킵니다.
    # def _idle(self, img):
    #     return img
    #
    # def flipImage(self, img, option_number):
    #     # 이미지 반전,  0:상하, 1 : 좌우로 반전합니다.
    #     img = cv2.flip(img, option_number)
    #     return img
    #
    # def rotateImage(self, image, angle):
    #     # 이미지 회전은 cv2.getRotationMatrix2D로 회전시킬 매트릭스를 생성한 다음 cv2.warpAffine으로 회전합니다.
    #     image_center = tuple(np.array(image.shape[1::-1]) / 2)
    #     rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    #     result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    #     return result
    #
    # # 옵션 난수를 받아서 augmentation을 실행합니다.
    # def _data_flip(self, img, augment_number):
    #
    #     if augment_number == 1:
    #         return self.flipImage(img, 1)
    #     elif augment_number == 2:
    #         return self.flipImage(img, 0)
    #     else:
    #         return self._idle(img)
    #
    # def _data_rotation(self, img, augment_number):
    #
    #     if augment_number == 1:
    #         return self.rotateImage(img, 5)
    #     elif augment_number == 2:
    #         return self.rotateImage(img, -5)
    #     elif augment_number == 3:
    #         return self.rotateImage(img, 10)
    #     elif augment_number == 4:
    #         return self.rotateImage(img, -10)
    #     else:
    #         return self._idle(img)
    #
    # def aug_percentage(self, percentage):
    #     flip_percentage_n = (2*percentage)+2
    #     rot_percentage_n = (4*percentage)+4
    #     return flip_percentage_n, rot_percentage_n
    #
    # def augmentation_reader(self, x_list, y_list, percentage):
    #     flip_per, rot_per = self.aug_percentage(percentage)
    #     random_number_for_flip = np.random.randint(0, flip_per, size=len(x_list))
    #     random_number_for_rotate = np.random.randint(0, rot_per, size=len(x_list))
    #
    #     aug_x_list = []
    #     aug_y_list = []
    #
    #     for i in range(len(random_number_for_flip)):
    #         x = self._data_rotation(self._data_flip(x_list[i], random_number_for_flip[i]), random_number_for_rotate[i])
    #         y = self._data_rotation(self._data_flip(y_list[i], random_number_for_flip[i]), random_number_for_rotate[i])
    #         aug_x_list.append(x)
    #         aug_y_list.append(y)
    #
    #     return aug_x_list, aug_y_list