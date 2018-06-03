"""
학습모듈로 메인모듈에서 옵션값들을 전달받아 데이터로더와 모델을 불러와서 실질적으로 학습을 진행하며 학습 중간에 밸리데이션을 진행하여 얼마나 학습이 진행되었는가 체크할 수 있습니다.

- 텐서보드로 모델과 성능을 확인해 볼 수 있습니다.

- 텐서보드 구동은 윈도우 기준 Anaconda prompt에서 tensorboard --logdir=로그파일경로 로 실행한 뒤 화면에 뜨는 주소창으로 접속하면 확인 할 수 있습니다.
  ex) tensorboard --logdir=C:\imsi
"""


import tensorflow as tf
import Portpolio.brain_tumor.unet.loader as loader
import Portpolio.brain_tumor.unet.unet as unet
import time
import os
import cv2
import numpy as np
import Portpolio.brain_tumor.unet.performance_eval as pe

# GPU가 여러개인경우 특정 GPU에서만 작업을 진행 할 수 있게 GPU 환경을 고정해줍니다.
# GPU가 n개일 경우 순차적으로 0번부터 n-1번까지 존재합니다.
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class Trainer:
    def __init__(self, model_root_channel, upsampling_mode, normalization_mode, training_data_path, validation_data_path, model_path, downsampling_option,
                 validation_percentage, initial_learning_rate, decay_step, decay_rate, epoch, img_size, n_class, batch_size, saving_epoch, loss, groupnormalization_n, channel_mode,
                 activation):

        # main.py에서 전달 받은 옵션 값
        self.training_path = training_data_path
        self.validation_path = validation_data_path
        self.model_path = model_path
        self.val_data_cnt = validation_percentage
        self.init_learning = initial_learning_rate
        self.decay_step = decay_step
        self.decay_rate = decay_rate
        self.epoch_num = epoch
        self.batch_size = batch_size
        self.upsampling_mode = upsampling_mode
        self.normalization_mode = normalization_mode
        self.activation = activation
        self.saving_epoch = saving_epoch
        self.loss = loss
        self.group_n = groupnormalization_n
        self.channel_mode = channel_mode
        self.downsampling_option = downsampling_option
        self.label_channel = n_class + 1
        self.model_root_channel = model_root_channel

        # 데이터로더 모듈 initialize
        self.data_loader = loader.DataLoader(img_size=img_size)

        print('')
        print('')
        print('')
        print('>>> Data Loading Started')
        print('')
        dstime = time.time()

        # 데이터로더 모듈로 학습데이터와 라벨데이터의 경로리스트와 데이터셋 개수를 가져옵니다
        self.trainX, self.trainY, self.data_count = self.data_loader.data_list_load(self.training_path, mode='train')
        self.valX, self.valY, self.val_data_count = self.data_loader.data_list_load(self.validation_path, mode='train')

        detime = time.time()
        print('>>> Data Loading Complete. Consumption Time :', detime - dstime)
        print('')
        print('>>> Dataset Split Started')
        print('')
        dsstime = time.time()
        dsetime = time.time()
        print('>>> Train Dataset Count:', len(self.trainX), 'valdation Dataset Count:', len(self.valX))
        print('')
        print('>>> Data Split Complete. Consumption Time :', dsetime - dsstime)
        print('')

        # 모델 모듈을 initialize 시키고 필요한 옵션값들을 전달 해 줍니다.
        #  n_channel은 학습데이터(X data)의 이미지 채널수를 입력합니다. 보통 Gray scaling을 하면 1채널, 안하면 RGB로 3채널입니다.
        self.model = unet.Model(upsampling_mode=self.upsampling_mode, normalization_mode=self.normalization_mode, img_size=img_size, n_channel=1, n_class=self.label_channel,
                                batch_size=self.batch_size, model_root_channel=self.model_root_channel, loss=self.loss, groupnormalization_n=self.group_n, channel_mode=self.channel_mode, activation=self.activation,
                                downsampling_option=self.downsampling_option)
        self.p_eval = pe.performance()

        # TB
        self.merged_summary = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter('./logs')

    # 학습을 진행하는 옵티마이저입니다. Adaptive Momentum(Adam) 방식을 사용합니다.
    def optimizer(self, global_step):
        exponential_decay_learning_rate = tf.train.exponential_decay(learning_rate=self.init_learning, global_step=global_step, decay_steps=self.decay_step, decay_rate=self.decay_rate, staircase=True, name='learning_rate')
        self.optimizer = tf.train.AdamOptimizer(learning_rate=exponential_decay_learning_rate).minimize(self.model.loss, global_step=global_step)

    # 학습을 진행하는 메소드입니다.
    def train(self):

        # 배치정규화를 진행하는 경우 배치정규화의 스탭을 결정하는 변수로 0입니다.
        global_step = tf.Variable(0, trainable=False)

        # 배치정규화를 진행하는 경우 배치별 이동평균과 표준편차를 갱신해주는 update operation을 실행하고 지정해줍니다.
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.optimizer(global_step)

        # 각각의 전체 데이터셋을 배치사이즈로 나누어 한 에폭당 몇 스텝이 진행되는가 계산합니다.
        train_step = int(len(self.trainX) / self.batch_size)
        val_step = int(len(self.valX) / self.batch_size)

        print('>>> Train step:', train_step, 'Validation step:', val_step)
        print('')

        # 텐서플로 그래프가 생성되고 작업이 수행되는 세션을 선언해줍니다.
        with tf.Session() as sess:

            # 모델을 저장하는 것은 모델의 가중치 같은 변수들을 저장하는 것인데 이를 텐서플로에선 ckpt, 체크포인트 파일이라고 합니다. ckpt 파일을 저장하는 텐서플로 클래스입니다.
            saver = tf.train.Saver()

            # 텐서보드에 생성한 모델의 그래프를 저장해줍니다.
            self.writer.add_graph(sess.graph)

            # 세션상의 글로벌 변수 초기화를 진행합니다. 변수 초기값은 개별 변수별로 지정해줘야합니다.
            sess.run(tf.global_variables_initializer())
            # saver.restore(sess, '/home/bjh/new_work/180424/dice,lr0.004,dr0.9,epoch200,batch32,channel32,resize,lrelu,NHWC.resizepool_neighbor-neighbor/init_model/Unet.ckpt')
            # print(">>> Model Loaded")
            print("BEGIN TRAINING")

            total_training_time = 0

            # 전달 받은 epoch 수 만큼 학습을 진행하는 loop 문 입니다.
            for epoch in range(self.epoch_num):

                # 최대수치 체크
                acc_list = []
                mean_iou_list = []
                unfiltered_iou_list = []
                loss_list = []

                # 컨퓨전 매트릭스
                confusion_list = []

                start = time.time()

                total_cost = 0
                total_val_iou = 0
                total_val_acc = 0
                total_val_unfiltered_iou = 0
                step = 0

                # 경로 내의 학습데이터를 마찬가지로 랜덤 셔플해줍니다. 경로 셔플과 마찬가지의 효과를 줍니다.
                trainX, trainY = self.data_loader.data_shuffle(self.trainX, self.trainY)

                # 한 에폭마다 학습하는 각 개별 스텝을 진행하는 loop 문 입니다.
                for idx in range(train_step):

                    # 데이터모듈을 이용하여 매 스탭마다 학습에 사용할 데이터 경로를 불러오며 스텝이 진행되면 다음 배치데이터를 불러옵니다.
                    batch_xs_list, batch_ys_list = self.data_loader.next_batch(data_list=trainX, label=trainY, idx=idx, batch_size=self.batch_size)

                    # 데이터모듈을 이용하여 위에서 불러온 데이터 경로에서 이미지 데이터를 읽어서 배치데이터를 만듭니다.
                    batch_xs = self.data_loader.read_image_grey_resized(batch_xs_list)
                    batch_ys = self.data_loader.read_label_grey_resized(batch_ys_list)

                    # 모델에 데이터를 넣어 줄 Feed Dict입니다.
                    tr_feed_dict = {self.model.X: batch_xs, self.model.Y: batch_ys, self.model.training: True, self.model.drop_rate: 0.2}
                    
                    # 모델을 위에서 선언한 session을 run 시켜서 학습시키고 결과물로 cost값을 받습니다.
                    cost, _ = sess.run([self.model.loss, self.optimizer], feed_dict=tr_feed_dict)

                    total_cost += cost
                    step += 1

                    # 학습 과정에서의 현재 에폭과 스텝 그리고 배치 Loss 값을 출력합니다.
                    print('Epoch:', '[%d' % (epoch + 1), '/ %d]  ' % self.epoch_num, 'Step:', step, '/', train_step, '  Batch loss:', cost)

                # 한 에폭마다 학습이 완료되고 해당 모델로 밸리데이션을 진행하는 loop 문 입니다.
                for idx in range(val_step):

                    # 데이터모듈을 이용하여 매 스탭마다 밸리데이션에 사용할 데이터 경로를 불러오며 스텝이 진행되면 다음 배치데이터를 불러옵니다.
                    val_batch_xs_list, val_batch_ys_list = self.data_loader.next_batch(data_list=self.valX, label=self.valY, idx=idx, batch_size=self.batch_size)

                    # 데이터모듈을 이용하여 위에서 불러온 데이터 경로에서 이미지 데이터를 읽어서 배치데이터를 만듭니다.
                    val_batch_xs = self.data_loader.read_image_grey_resized(val_batch_xs_list)
                    val_batch_ys = self.data_loader.read_label_grey_resized(val_batch_ys_list)

                    # 모델에 데이터를 넣어 줄 Feed Dict입니다.
                    val_feed_dict = {self.model.X: val_batch_xs, self.model.Y: val_batch_ys, self.model.training: False, self.model.drop_rate: 0}

                    # 밸리데이션 결과 IoU(Intersection of Union)을 계산합니다. Image Segmentation에선 IoU를 보통 Accuracy로 사용합니다.
                    # model.iou에선 [acc, mean_iou, unfiltered_iou]를 리턴합니다.
                    val_results, predicted_result = sess.run([self.model.results, self.model.foreground_predicted], feed_dict=val_feed_dict)
                    # acc, val_mean_iou, val_unfiltered_iou = val_results

                    # 받은 배치 IoU값을 리스트로 변환합니다.
                    ious = list(val_results[0])
                    accs = list(val_results[1])
                    # print(accs)  # [ 0.  5.  2. 15.  0.  7.  0. 12.  0.  5.  8.  0. 17.  0.  0. 10.  0. 29. 0.  9. 11.  0.  0. 13.  0. 15. 22.  3. 14.  0.]

                    #비정상-오브젝트있음 : TruePositive / 비정상-오브젝트없음 : FalseNegative / 정상-오브젝트있음 : FalsePositive / 정상-오브젝트없음 : TrueNegative
                    val_confusion_list = []
                    for idx, acc in enumerate(accs):
                        if 'abnorm' in val_batch_xs_list[idx] and acc != 0.:
                            confusion_list.append('TP')
                            val_confusion_list.append('TP')
                        elif 'abnorm' in val_batch_xs_list[idx] and acc == 0.:
                            confusion_list.append('FN')
                            val_confusion_list.append('FN')
                        elif 'norm' in val_batch_xs_list[idx] and acc != 0.:
                            confusion_list.append('FP')
                            val_confusion_list.append('FP')
                        elif 'norm' in val_batch_xs_list[idx] and acc == 0.:
                            confusion_list.append('TN')
                            val_confusion_list.append('TN')

                    #TP와 TN을 count해서 정확도를 구함
                    TP_cnt = val_confusion_list.count('TP')
                    TN_cnt = val_confusion_list.count('TN')
                    val_tot_data_cnt = len(val_confusion_list)
                    val_batch_acc = (TP_cnt + TN_cnt) / val_tot_data_cnt

                    # 진단 정확도 판단을 위해 전체 IoU 의 길이를 받아냅니다. IoU가 0이거나 0에 매우 가까우면 제대로 진단을 하지 못했다고 판단 할 수 있습니다.
                    # 따라서 IoU가 기준치 이상인 값들만 추려내면 질병이 존재한다고 진단 할 수 있습니다.
                    # before_filtered_length = len(ious)

                    # 전체 평균 IoU를 계산합니다.
                    unfiltered_iou = np.mean(ious)

                    # IoU가 0.02 이상, 즉 일정이상 예측해낸 IoU 값들만 모아서 진단 정확도와 질병으로 판단 했을 때의 IoU 값을 따로 계산합니다.
                    iou_list = []

                    for iou in ious:
                        if iou > 0.01:
                            iou_list.append(iou)

                    if len(iou_list) == 0:
                        mean_iou = 0
                    else:
                        mean_iou = np.mean(iou_list)

                    # 배치별 IoU값과 정확도를 전체 IoU값과 정확도에 더합니다. 에폭이 종료되면 평균 IoU와 평균 정확도로 환산합니다.
                    total_val_acc += val_batch_acc
                    total_val_iou += mean_iou
                    total_val_unfiltered_iou += unfiltered_iou

                    # todo : 데이터분할 바뀜 : 여기 변경(아래 다 주석. 이미지 저장 안함)
                    # # 학습 시작 에폭과 끝에폭 그리고 saving epoch의 배수마다 이미지와 모델을 저장하게 합니다.
                    # if (epoch+1) % self.saving_epoch == 0 or epoch+1 == self.epoch_num or epoch == 0:
                    #
                    #     # 밸리데이션 결과 이미지를 저장하는 경로입니다.
                    #     # val_img_save_path 는 학습이미지(원본이미지)와 예측이미지를 Overlap 시켜 환부에 마스크 이미지를 씌워주며
                    #     # raw_val_img_save_path는 예측이미지를, label_val_img_save_path는 라벨이미지를 저장하는 경로입니다.
                    #     # 윈도우에서 실행시키려면 /를 \\로 교체해야합니다.
                    #     val_img_save_path = './imgs/' + str(epoch+1) + '/merged'
                    #     raw_val_img_save_path = './imgs/' + str(epoch + 1) + '/pred'
                    #     label_val_img_save_path = './imgs/' + str(epoch + 1) + '/label'
                    #
                    #     # 각 개별 경로가 존재하는지 확인하고 없는 경우 경로를 생성합니다.
                    #     if not os.path.exists(val_img_save_path):
                    #         os.makedirs(val_img_save_path)
                    #
                    #     if not os.path.exists(raw_val_img_save_path):
                    #         os.makedirs(raw_val_img_save_path)
                    #
                    #     if not os.path.exists(label_val_img_save_path):
                    #         os.makedirs(label_val_img_save_path)
                    #
                    #     # 예측된 배치 결과를 loop하면서 개별 이미지로 저장하는 loop문입니다.
                    #     for img_idx, label in enumerate(predicted_result):
                    #
                    #         # 각 이미지 종류별 이미지를 저장하는 절대경로입니다. 이미지 파일명은 '밸리데이션 index'_'이미지 번호'.png 로 저장됩니다.
                    #         val_img_fullpath = val_img_save_path \
                    #                            + '/' + val_batch_xs_list[img_idx].split(os.path.sep)[-5] \
                    #                            + '_' + val_batch_xs_list[img_idx].split(os.path.sep)[-4] \
                    #                            + '_' + val_batch_xs_list[img_idx].split(os.path.sep)[-1]
                    #
                    #         raw_val_img_fullpath = raw_val_img_save_path \
                    #                                + '/' + val_batch_xs_list[img_idx].split(os.path.sep)[-5] \
                    #                                + '_' + val_batch_xs_list[img_idx].split(os.path.sep)[-4] \
                    #                                + '_' + val_batch_xs_list[img_idx].split(os.path.sep)[-1]
                    #
                    #         label_val_img_fullpath = label_val_img_save_path \
                    #                                  + '/' + val_batch_xs_list[img_idx].split(os.path.sep)[-5] \
                    #                                  + '_' + val_batch_xs_list[img_idx].split(os.path.sep)[-4] \
                    #                                  + '_' + val_batch_xs_list[img_idx].split(os.path.sep)[-1]
                    #
                    #         # 라벨이미지를 가져옵니다.
                    #         test_image = val_batch_xs[img_idx]
                    #
                    #         # 라벨이미지 저장을 위해 3채널 RGB 데이터가 필요하고 배치 차원을 맞춰주기 위해 차원확장을 진행합니다.
                    #         # 이미지의 차원은 현재 [B, H, W, C] 로 배치, 세로, 가로, 채널로 되어있습니다.
                    #         test_image = np.expand_dims(test_image, axis=0)
                    #
                    #         # 예측 결과를 threshold(기준 값을 경계로 0과 1 바이너리화를 진행합니다.)
                    #         # 사용법 : _, img = cv2.threshold(이미지, 경계값, 바이너리최대값, 바이너리옵션)으로
                    #         # 옵션을 cv2.THRESH_BINARY로 진행하면 검은색 흰색 이미지가, cv2.THRESH_BINARY_INV로 진행하면 흰색 검은색 이미지가 저장됩니다.
                    #         # 자세한 사항은 cv2 홈페이지를 참조하세요.
                    #         _, pred_image = cv2.threshold(label, 0.5, 1.0, cv2.THRESH_BINARY)
                    #
                    #         # cv2의 결과는 2차원(H, W) 입니다. 따라서 마찬가지로 0차원과 4차원에 차원을 덧대주어서 차원을 맞춰줍니다.
                    #         pred_image = np.expand_dims(pred_image, axis=3)
                    #         pred_image = np.expand_dims(pred_image, axis=0)
                    #
                    #         # 예측이미지의 마스크 색을 결정합니다.
                    #         # 예측이미지값을 R에 넣으면 빨간 마스킹 이미지가, B에 넣으면 파란 마스킹 이미지가, G에 넣으면 녹색 마스킹 이미지가 생성됩니다.
                    #         G = np.zeros([1, 256, 256, 1])
                    #         B = np.zeros([1, 256, 256, 1])
                    #         R = pred_image
                    #
                    #         # R, G, B 채널을 concat 해서 하나의 차원에 정렬해줍니다.
                    #         pred_image = np.concatenate((B, G, R), axis=3)
                    #
                    #         # 필요없는 차원을 squeeze 해줍니다.
                    #         pred_image = np.squeeze(pred_image)
                    #
                    #         # test_image는 원본이 그대로 필요하므로 R, G, B 모두에 데이터를 넣어줍니다.
                    #         tR = test_image
                    #         tG = test_image
                    #         tB = test_image
                    #
                    #         # 위 concat, squeeze와 동일합니다
                    #         test_image = np.concatenate((tB, tG, tR), axis=3)
                    #         test_image = np.squeeze(test_image)
                    #
                    #         # 위 과정을 label_img도 동일하게 진행해줍니다.
                    #         label_image = val_batch_ys[img_idx][:,:,0]
                    #         label_image = np.expand_dims(label_image, axis=0)
                    #         label_image = np.expand_dims(label_image, axis=3)
                    #
                    #         lR = label_image
                    #         lG = label_image
                    #         lB = label_image
                    #
                    #         label_image = np.concatenate((lB, lG, lR), axis=3)
                    #         label_image = np.squeeze(label_image)
                    #
                    #         # 바이너리화된 이미지는 (0, 1)의 데이터 이므로 RGB로 변경하려면 255를 곱해주어야 합니다.
                    #         label_image = label_image * 255
                    #         cv2.imwrite(label_val_img_fullpath, label_image)
                    #
                    #         # 위와 동일합니다.
                    #         test_image = test_image.astype(float)
                    #         pred_image = pred_image * 255
                    #         cv2.imwrite(raw_val_img_fullpath, pred_image)
                    #
                    #         # 원본이미지에 예측결과를 마스킹해줍니다.
                    #         # 마스킹 비율을 결정하는 파라메터가 w이고 각 이미지의 적용비율은 p로 결정합니다. w와 p를 바꿔가면서 저장하며 가시성 좋은 값을 찾으면 됩니다.
                    #         w = 40
                    #         p = 0.0001
                    #         result = cv2.addWeighted(pred_image, float(100 - w) * p, test_image, float(w) * p, 0)
                    #         cv2.imwrite(val_img_fullpath, result * 255)

                # 모델을 저장할 경로를 확인하고 없으면 만들어줍니다.
                if os.path.exists(self.model_path + '/' + str(epoch + 1)) is False:
                    os.makedirs(self.model_path + '/' + str(epoch + 1))
                # 모델 저장을 위한 절대경로입니다. '파일명'.ckpt로 저장합니다.
                save_path = self.model_path + '/' + str(epoch + 1) + '/Unet.ckpt'
                # 모델을 저장합니다.
                saver.save(sess, save_path)
                print(">>> Model SAVED")
                print('')

                end = time.time()
                training_time = end - start
                total_training_time += training_time

                Loss = total_cost / train_step
                Valdation_IoU = total_val_iou / val_step
                Valdation_Unfiltered_IoU = total_val_unfiltered_iou / val_step
                Valdation_Accuracy = total_val_acc / val_step

                print('Epoch:', '[%d' % (epoch + 1), '/ %d]  ' % self.epoch_num,
                      'Loss =', '{:.4f}  '.format(Loss),
                      'Valdation IoU:{:.4f}   '.format(Valdation_IoU),
                      'Valdation Unfiltered IoU:{:.4f}   '.format(Valdation_Unfiltered_IoU),
                      'Valdation Accuracy:{:.4f}   '.format(Valdation_Accuracy),
                      'Training time: {:.2f}  '.format(training_time))

                result_dict = {self.p_eval.acc: Valdation_Accuracy, self.p_eval.mean_iou: Valdation_IoU, self.p_eval.tot_iou: Valdation_Unfiltered_IoU, self.p_eval.loss:Loss}

                # TB
                summary = sess.run(self.merged_summary, feed_dict=result_dict)
                self.writer.add_summary(summary, global_step=epoch)

                acc_list.append(Valdation_Accuracy)
                mean_iou_list.append(Valdation_IoU)
                unfiltered_iou_list.append(Valdation_Unfiltered_IoU)
                loss_list.append(Loss)

                total_TP = confusion_list.count('TP')
                total_FN = confusion_list.count('FN')
                total_FP = confusion_list.count('FP')
                total_TN = confusion_list.count('TN')

                with open("/mnt/sdb/bjh/tr_result/dice_lr0.005_decay2500_dr0.9_batch28_rc32_prelu_original/model/train_result_epoch" + str(epoch+1) +".txt", "w", newline='\n') as f:
                    f.write('|  * ACC :{:.4f}'.format(np.max(acc_list)) + "\n")
                    f.write("|" + "\n")
                    f.write('|  * MEAN IOU :{:.4f}'.format(np.max(mean_iou_list)) + "\n")
                    f.write("|" + "\n")
                    f.write('|  * TOTAL IOU :{:.4f}'.format(np.max(unfiltered_iou_list)) + "\n")
                    f.write("|" + "\n")
                    f.write('|  * MIN LOSS :{:.4f}'.format(np.min(loss_list)) + "\n")
                    f.write("|" + "\n")
                    f.write('|  * VALIDATION CONFUSION MATRIX' + "\n")
                    f.write("|" + "\n")
                    f.write('|                     |  Predict True  |  Predict False' + "\n")
                    f.write('|   ------------------+----------------+--------------------' + "\n")
                    f.write('|   Groudtruth True   |      {0:^4}      |      {1}'.format(total_TP, total_FN) + "\n")
                    f.write("|   Groudtruth False  |      {0:^4}      |      {1}".format(total_FP, total_TN) + "\n")
                    f.write("|" + "\n")
                    f.write('|  * VALIDATION SENSITIVITY : {:.4f}'.format(total_TP / (total_TP + total_FN + 1e-6)) + "\n")
                    f.write("|" + "\n")
                    f.write('|  * VALIDATION SPECIFICITY : {:.4f}'.format(total_TN / (total_FP + total_TN + 1e-6)) + "\n")
                    f.write("|" + "\n")
                    f.write("=================================================================")

            print("")
            print("TRAINING COMPLETE")

