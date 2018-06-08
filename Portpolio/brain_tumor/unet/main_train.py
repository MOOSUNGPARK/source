"""
2D U-net Based Image Segmentation Deep-learning module.
'main_train.py' for train run,
'main_test.py' for test run,
'Unet.py' for model,
'loader.py' for data load pipeline,
'utils.py' for utility functions used at 'Unet.py',
'performance_eval.py' for tensorboard,
'Trainer.py' for training pipeline,
'Tester.py' for testing pipeline.

'main_train.py' is main python file for inputting hyper parameter used at 'Trainer.py'

- 학습 데이터는 의료영상데이터 기준으로 root폴더/환자폴더/img폴더/x 안에 학습이미지가, root폴더/환자폴더/img폴더/y 안에 라벨이미지가 있는 경로구성으로 되어있습니다.
  ex) 학습이미지(X data) : /home/bjh/new_train_img_label_filtered/11/img/x  , 라벨이미지(Y data) : /home/bjh/new_train_img_label_filtered/11/img/y
- Train dataset folder

- Train data

- 또한 리눅스와 윈도우는 경로구분자 기준이 다르므로 리눅스는 '/' 를 경로구분자로 사용하고 윈도우는 '\' 를 경로구분자로 사용합니다.
  다만 '\' 구분자는 정규표현식에서 사용하므로 \p와 같은 경로는 우선적으로 정규표현식으로 인식하므로 \를 2개 붙여줘야 합니다.
  ex) D:\\Results\\n\\1\\merged\\1_1.png

- 학습데이터와 라벨데이터의 파일명은 동일하게 해야하며 이미지의 상위폴더 x, y로 구분합니다.

- 학습된 모델은 '메인파일을 실행시킨 경로/model/에폭' 하에 저장됩니다. 값을 변경하면 몇 에폭마다 모델을 저장할 지 정할 수 있습니다. 이 값은 Trainer.py에서 변경합니다.

- Trainer 옵션 설명
1) training_data_path : 학습데이터의 root 경로입니다.

2) model_path : 모델을 저장할 경로입니다. 디폴트값인 './model'로 두면 main.py를 실행시킨 경로 아래의 model 폴더에 에폭별로 저장됩니다.

3) validation_percentage : 학습데이터와 밸리데이션데이터의 분할비율입니다. 정수만 입력해야하며 10을 입력하면 학습데이터의 10%를 밸리데이션 데이터로 이용합니다. 밸리데이션 데이터는 학습하지 않습니다.
                           머신러닝에서 데이터 분할은 보통 전체 데이터셋을 학습데이터(Train Dataset), 밸리데이션데이터(Validation Dataset), 테스트데이터(Test Dataset)으로 구분하며 일반적으로 6:2:2 비율이나 6:1:3, 7:0:3 비율로 분할합니다.

4) initial_learning_rate : 하이퍼파라메터인 Learning Rate의 초기값입니다. 학습율이라고도 하며 각 단계별로 얼마만큼 가중치를 갱신할 지 정해줍니다. Learning rate는 step별로 감소(decay)시켜 수렴에 도움을 줄 수 있습니다.

5) decay_step : Learning rate의 감소 스텝을 결정합니다. decay_step은 몇 스탭마다 Learning rate를 감소시킬지 결정합니다.

6) decay_rate : decay_rate는 얼마의 비율로 Learning rate를 감소시킬지 결정합니다.
                ex) learning rate = 0.95, decay step = 1000, decay rate = 0.9 -> 매 1000스텝마다 learning rate에 0.9를 곱합니다. (10%씩 감소시킵니다)

7) epoch : 학습을 얼마나 할 지 결정합니다. 보통 1에폭은 전체데이터셋을 한번 학습하는 과정을 의미하며 1스텝은 배치 하나를 학습하는 것을 의미합니다.
           따라서 전체데이터셋/배치단위 = 총 스텝 수이며 총 스텝이 다 돌면 1에폭이 돌았다고 할 수 있습니다.

8) img_size : 이미지 데이터셋의 크기가 다를 경우 공통적으로 어느 사이즈로 만들어 줄 지 결정합니다. resizing 사이즈로 픽셀사이즈입니다.

9) n_class : 라벨이미지가 몇개의 클래스인지 결정합니다.

10) batch_size : 배치사이즈를 결정합니다. GPU의 VRAM에 따라서 OOM(Out Of Memory: VRAM 메모리 부족)이 뜨지 않도록 배치사이즈와 이미지사이즈를 결정해줘야합니다.

11) upsampling_mode : Unet 모델의 업샘플링 방법을 결정합니다. 'Resize'를 입력하면 Resize Convolution으로 업샘플링을 진행하고 'Transpose'을 입력하면 Deconvolution(Transpose convolution)으로 업샘플링을 진행합니다.

12) normalization_mode : Unet 모델의 정규화 방법을 결정합니다. 'Batch'를 입력하면 Batch Normalization을 실행하고 'Group'을 입력하면 Group Normalization을 실행하고 'None'을 입력하면 정규화를 진행하지 않습니다.

13) model_root_channel : Unet 모델의 첫 번째 레이어의 Feature map 개수를 정해줍니다. Batch size와 Root channel 수를 조절하여 VRAM에 따라서 OOM이 뜨지 않게 조절해줍니다. 채널수가 많아지면 파라메터의 양이 늘어납니다.

14) activation : 활상화함수를 결정합니다. relu, lrelu, prelu, elu, selu 등을 선택할 수 있습니다.

15) augmentation_mode : 데이터로더에서 학습 시 data augmentation을 수행 할 지 안할 지 결정합니다. True로 두면 수행합니다.
                        Data augmentation은 좌우 Flip, 상하 Flip, +- 10도와 +- 20도의 Rotation이 구현되어있습니다.

16) saving_epoch : 학습 과정에서 몇 에폭마다 밸리데이션 결과와 모델을 저장할 지 결정합니다. 5를 입력하면 5에폭마다 결과를 저장합니다.

17) augmentation_percentage : normal 데이터와 augmentationed 데이터의 비율을 결정합니다. 정수로 입력합니다.
                              ex) augmentation_percentage = 3  -> normal data 3 : augmentationed data 1

18) background_label : background에 라벨링된 데이터를 자동으로 생성하여 사용할 지 안할 지 결정합니다. True로 두면 수행합니다.

19) loss : 모델에서 사용 할 Loss function을 결정합니다. 옵션으로 'dice', 'focal', 'cross_entropy'를 정할 수 있습니다.

20) groupnormalization_n : Group Normalization을 사용하는 경우 Group 개수를 지정해줍니다.

21) channel_mode : 배치데이터의 shape를 결정합니다. NCHW, NHWC 두가지 옵션을 사용할 수 있습니다.
                   https://www.tensorflow.org/performance/performance_guide#data_formats 를 참조하세요.
"""


from Portpolio.brain_tumor.unet.Trainer import Trainer

if __name__ == "__main__":

    full_data_path = '/home/bjh/aneurysm_new_data_train'
    label_only_data_path = '/home/bjh/new_train_img_label_only'
    filtered_label_only_data_path = '/home/bjh/new_train_img_label_filtered/abnorm'

    tr_seperated_data_path = 'D:\\dataset\\Brain_Aneurysm_new_dataset\\train\\'
    val_seperated_data_path = 'D:\\dataset\\Brain_Aneurysm_new_dataset\\test\\'

    # unet_trainer = Trainer(training_data_path=[filtered_label_only_data_path], model_path='./model', validation_percentage=10,
    #                        initial_learning_rate=0.004, decay_step=2500, decay_rate=0.9, epoch=200, img_size=256,
    #                        n_class=1, batch_size=32, upsampling_mode='resize', normalization_mode='Batch', model_root_channel=32, activation='lrelu',
    #                        augmentation_mode=False, saving_epoch=5, augmentation_percentage=None, background_label=True,
    #                        loss='dice', groupnormalization_n=None, channel_mode='NHWC')

    unet_trainer = Trainer(training_data_path=[tr_seperated_data_path],
                           validation_data_path=[val_seperated_data_path],
                           model_path='./model',
                           validation_percentage=10,
                           initial_learning_rate=0.005,
                           decay_step=2500,
                           decay_rate=0.9,
                           epoch=400,
                           img_size=256,
                           n_class=1,
                           batch_size=28,
                           upsampling_mode='transpose',  # resize, transpose
                           normalization_mode='batch',  # batch, group
                           model_root_channel=32,
                           activation='prelu',  # lrelu, prelu, selu, elu, relu, None
                           saving_epoch=2,
                           loss='dice_sum',  # dice, focal, huber, cross_entropy, dice_sum
                           groupnormalization_n=None,
                           channel_mode='NHWC',
                           downsampling_option='maxpool')  # linear, neighbor, maxpool

    unet_trainer.train()
