"""
학습이 완료된 모델로 테스트를 진행 할 수 있는 테스트 모듈입니다.


"""

import tensorflow as tf
import loader
import unet
import time
import os
import cv2
import numpy as np

# GPU가 여러개인경우 특정 GPU에서만 작업을 진행 할 수 있게 GPU 환경을 고정해줍니다.
# GPU가 n개일 경우 순차적으로 0번부터 n-1번까지 존재합니다.
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


class Tester:
    def __init__(self, test_data_path, model_path, batch_size, img_size, upsampling_mode, downsampling_option, normalization_mode, label_channel, model_root_channel):

        # main.py에서 전달 받은 옵션 값
        self.test_path = test_data_path
        self.model_path = model_path
        self.batch_size = batch_size
        self.img_size = img_size
        self.upsampling_mode = upsampling_mode
        self.downsampling_option = downsampling_option
        self.normalization_mode = normalization_mode
        self.n_channel = 1
        self.label_channel = label_channel
        self.model_root_channel = model_root_channel
        self.ckpt_path = self.model_path
        # 데이터로더 모듈 initialize
        self.data_loader = loader.DataLoader(img_size=img_size)

        # 데이터로더 모듈로 학습데이터와 라벨데이터의 경로리스트와 데이터셋 개수를 가져옵니다
        self.img_list, self.data_count = self.data_loader.data_list_load(self.test_path, mode='test')

        self.model = unet.Model(upsampling_mode=self.upsampling_mode, downsampling_option=self.downsampling_option,
                                normalization_mode=self.normalization_mode, img_size=img_size,
                                n_channel=1, n_class=self.label_channel,
                                batch_size=self.batch_size, model_root_channel=self.model_root_channel)

    def test(self):
        # tf.reset_default_graph()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, self.ckpt_path)
            print('model restored')

            test_step = int(len(self.img_list) / self.batch_size)

            for idx in range(test_step):
                # 데이터모듈을 이용하여 매 스탭마다 밸리데이션에 사용할 데이터 경로를 불러오며 스텝이 진행되면 다음 배치데이터를 불러옵니다.
                # todo data_loader trainer의 validation 함수 가져오기
                test_batch_xs_list = self.data_loader.next_batch_test(data_list=self.img_list, idx=idx, batch_size=self.batch_size)
                # 데이터모듈을 이용하여 위에서 불러온 데이터 경로에서 이미지 데이터를 읽어서 배치데이터를 만듭니다.
                test_batch_xs = self.data_loader.read_image_grey_resized(data_list=test_batch_xs_list)
                # 모델에 데이터를 넣어 줄 Feed Dict입니다.
                test_feed_dict = {self.model.X: test_batch_xs, self.model.training: False,self.model.drop_rate: 0}

                predicted_result = sess.run(self.model.foreground_predicted, feed_dict=test_feed_dict)

                # 윈도우에서 실행시키려면 /를 \\로 교체해야합니다.
                test_img_save_path = './imgs_test_result/results'

                # print('pre: ', str(len(predicted_result)))
                # print('t_batch: ', str(len(test_batch_xs_list)))
                # 예측된 배치 결과를 loop하면서 개별 이미지로 저장하는 loop문입니다.
                for img_idx, label in enumerate(predicted_result):
                    # 각 이미지 종류별 이미지를 저장하는 절대경로입니다. 이미지 파일명은 '밸리데이션 index'_'이미지 번호'.png 로 저장됩니다.
                    test_img_path = test_img_save_path \
                                        + '/' + test_batch_xs_list[img_idx].split(os.path.sep)[-5] \
                                        + '/' + test_batch_xs_list[img_idx].split(os.path.sep)[-4] \
                                        + '/merged/'
                    pred_test_img_path = test_img_save_path \
                                        + '/' + test_batch_xs_list[img_idx].split(os.path.sep)[-5] \
                                        + '/' + test_batch_xs_list[img_idx].split(os.path.sep)[-4] \
                                        + '/predicted/'

                    # 각 개별 경로가 존재하는지 확인하고 없는 경우 경로를 생성합니다.
                    if not os.path.exists(test_img_path):
                        os.makedirs(test_img_path)
                    if not os.path.exists(pred_test_img_path):
                        os.makedirs(pred_test_img_path)

                    test_img_fullpath = test_img_path + test_batch_xs_list[img_idx].split(os.path.sep)[-1]
                    pred_test_img_fullpath = pred_test_img_path + test_batch_xs_list[img_idx].split(os.path.sep)[-1]

                    print('test file path: ', test_img_fullpath)
                    test_image = test_batch_xs[img_idx]

                    # 이미지 저장을 위해 3채널 RGB 데이터가 필요하고 배치 차원을 맞춰주기 위해 차원확장을 진행합니다.
                    # 이미지의 차원은 현재 [B, H, W, C] 로 배치, 세로, 가로, 채널로 되어있습니다.
                    test_image = np.expand_dims(test_image, axis=0)

                    # 예측 결과를 threshold(기준 값을 경계로 0과 1 바이너리화를 진행합니다.)
                    # 사용법 : _, img = cv2.threshold(이미지, 경계값, 바이너리최대값, 바이너리옵션)
                    # 바이너리옵션을 cv2.THRESH_BINARY로 진행하면 검은색 흰색 이미지가, cv2.THRESH_BINARY_INV로 진행하면 흰색 검은색 이미지가 저장됩니다.
                    # 자세한 사항은 cv2 홈페이지를 참조하세요.
                    _, pred_image = cv2.threshold(label, 0.5, 1.0, cv2.THRESH_BINARY)

                    # cv2의 결과는 2차원(H, W) 입니다. 따라서 마찬가지로 0차원과 4차원에 차원을 덧대주어서 차원을 맞춰줍니다.
                    pred_image = np.expand_dims(pred_image, axis=3)
                    pred_image = np.expand_dims(pred_image, axis=0)

                    # 예측이미지의 마스크 색을 결정합니다. 예측이미지값을 R에 넣으면 빨간 마스킹 이미지가, B에 넣으면 파란 마스킹 이미지가, G에 넣으면 녹색 마스킹 이미지가 생성됩니다.
                    G = np.zeros([1, 256, 256, 1])
                    B = np.zeros([1, 256, 256, 1])
                    R = pred_image

                    # R, G, B 채널을 concat 해서 하나의 차원에 정렬해줍니다.
                    pred_image = np.concatenate((B, G, R), axis=3)

                    # 필요없는 차원을 squeeze 해줍니다.
                    pred_image = np.squeeze(pred_image)

                    # test_image는 원본이 그대로 필요하므로 R, G, B 모두에 데이터를 넣어줍니다.
                    tR = test_image
                    tG = test_image
                    tB = test_image

                    # 위 concat, squeeze와 동일합니다
                    test_image = np.concatenate((tB, tG, tR), axis=3)
                    test_image = np.squeeze(test_image)
                    test_image = test_image.astype(float)
                    pred_image = pred_image * 255
                    cv2.imwrite(pred_test_img_fullpath, pred_image)

                    # 원본이미지에 예측결과를 마스킹해줍니다. 마스킹 비율을 결정하는 파라메터가 w이고 각 이미지의 적용비율은 p로 결정합니다.
                    # w와 p를 바꿔가면서 저장하며 가시성 좋은 값을 찾으면 됩니다.
                    w = 40
                    p = 0.0001
                    result = cv2.addWeighted(pred_image, float(100 - w) * p, test_image, float(w) * p, 0)
                    cv2.imwrite(test_img_fullpath, result * 255)

                # print('valdation IoU:{:.4f}   '.format(total_val_iou / val_step),
                #       'valdation Unfiltered IoU:{:.4f}   '.format(total_val_unfiltered_iou / val_step),
                #       'valdation Accuracy:{:.4f}   '.format(total_val_acc / val_step),
                #       )
