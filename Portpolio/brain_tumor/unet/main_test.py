"""
2D U-net을 기반으로한 이미지 세그먼테이션 테스트 모듈입니다. main_test.py로 구동을 하며 Tester.py가 학습모듈, Unet.py가 모델모듈, loader.py가 데이터로딩모듈, utils.py가 모델에서 사용하는 함수들이 담긴 모듈입니다.

main_test.py는 학습을 실행시키는 메인 파일로 파일 경로와 기초적인 하이퍼 파라메터의 값을 넣어줍니다.

- 학습 데이터는 의료영상데이터 기준으로 root폴더/환자폴더/img폴더/x 안에 학습이미지가, root폴더/환자폴더/img폴더/y 안에 라벨이미지가 있는 경로구성으로 되어있습니다.
  ex) 학습이미지(X data) : /home/bjh/new_train_img_label_filtered/11/img/x  , 라벨이미지(Y data) : /home/bjh/new_train_img_label_filtered/11/img/y

- 또한 리눅스와 윈도우는 경로구분자 기준이 다르므로 리눅스는 '/' 를 경로구분자로 사용하고 윈도우는 '\' 를 경로구분자로 사용합니다.
  다만 '\' 구분자는 정규표현식에서 사용하므로 \p와 같은 경로는 우선적으로 정규표현식으로 인식하므로 \를 2개 붙여줘야 합니다.
  ex) D:\\Results\\n\\1\\merged\\1_1.png

- 학습데이터와 라벨데이터의 파일명은 동일하게 해야하며 이미지의 상위폴더 x로 구분합니다.

- Tester 옵션 설명

1) test_data_path : 테스트 데이터의 root 경로입니다.

2) model_path : 모델이 저장되어있는 경로입니다.

3) batch_size : 배치사이즈를 결정합니다. GPU의 VRAM에 따라서 OOM(Out Of Memory: VRAM 메모리 부족)이 뜨지 않도록 배치사이즈와 이미지사이즈를 결정해줘야합니다.

4) img_size : 출력 이미지의 픽셀사이즈입니다.

5) upsampling_mode : 불러올 Unet 모델의 업샘플링 방법을 입니다.

6) normalization_mode : 불러올 Unet 모델의 정규화 방법을 입니다.

7) n_class : 불러올 모델의 라벨이미지의 클래스 입니다.

8) model_root_channel : 불러올 Unet 모델의 첫 번째 레이어의 Feature map 개수입니다.

"""


from Portpolio.brain_tumor.unet.Tester import Tester

if __name__ == "__main__":

    test_data_path = '/home/bjh/new_train_img_label_filtered'
    model_path = './model/400/Unet.ckpt'

    unet_tester = Tester(test_data_path=[test_data_path], model_path=model_path, batch_size=28, img_size=256,
                         upsampling_mode='transpose', downsampling_option='maxpool', normalization_mode='batch',
                         label_channel=2, model_root_channel=32)

    unet_tester.test()
