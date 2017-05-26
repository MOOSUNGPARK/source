
import math               # 엔트로피 계산에 로그함수를 쓰기 위해 math 모듈을 불러온다.

import collections        # 분석 컬럼별 확률을 계산하기 위해 만드는 class_probabilities를 함수로 만들기 위해 사용하는 모듈을 불러온다.
                          # 자세히 설명하면 collection.Counter 함수를 사용하기 위함인데
                          # collection.Counter() 함수는 Hashing(책 p.110 참조) 할 수 있는 오브젝트들을
                          # count 하기 위해 서브 클래스들을 딕셔너리화 할 때 사용한다.

from functools import partial         # c5.0(엔트로피를 사용하는 알고리즘) 알고리즘 기반의 의사결정트리를
                                      # 제작하는 트리제작함수인 build_tree_id3에서 사용하는데
                                      # functools.partial() 함수는 기준 함수로 부터 정해진 기존 인자가 주어지는 여러개의 특정한 함수를 만들고자 할 때 사용한다.
                                      # http://betle.tistory.com/entry/PYTHON-decorator를 참조

from collections import defaultdict         # 팩토리 함수는 특정 형의 데이터 항목을 새로 만들기 위해 사용되는데
                                            # defaultdict(list) 함수는 missing value(결측값)을 처리 하기 위한
                                            # 팩토리 함수를 불러오는 서브 클래스를 딕셔너리화 할 떄 사용한다.


def entropy(class_probabilities):            # 해당 컬럼의 확률을 입력받아 해당 클래스의 엔트로피를 계산하는 함수
    return sum(-p * math.log(p, 2)           # 정보획득량을 계산하기 위한 조건부 엔트로피를 합산하는 수식
               for p in class_probabilities if p)        # 해당 컬럼에 빈도수가 존재하지 않으면 0인데 이경우 조건부 확률도 0이되게 되고
                                                         # 그럴 경우 log함수에서 0을 입력하면 에러가 발생한다.
                                                         # 따라서 0이 아닌 경우에만 엔트로피를 계산한다.

def class_probabilities(labels):       # 엔트로피 계산에 사용하는 컬럼의 확률을 계산하는 함수
    total_count = len(labels)          # 전체 count 수 = 해당 컬럼의 총 길이
                                       # 분석에 사용하는 데이터 구조가 다음과 같기 때문에 전체 count수는 len(labels)로 할 수 있다.
                                       # 데이터셋 = [ ( {데이터가 되는 컬럼의 키와 값으로 구성된 딕셔너리}, 분석타겟컬럼의 값  ) , ..... ]

    return [count / total_count for count in collections.Counter(labels).values()]      #계산한 확률을 list형으로 반환


def data_entropy(labeled_data):         # 전체 데이셋의 엔트로피
    labels = [label for _, label in labeled_data]
    probabilities = class_probabilities(labels)
    return entropy(probabilities)


def partition_entropy(subsets):         # 파티션된 노드들의 엔트로피
    total_count = sum(len(subset) for subset in subsets)        # subset은 라벨이 있는 데이터의 리스트의 리스트이다. 이것에 대한 엔트로피를 계산한다.
    return sum(data_entropy(subset) * len(subset) / total_count for subset in subsets)

#########################   데이터셋   #################################
# 데이터셋 = [ ( {데이터가 되는 컬럼의 키와 값으로 구성된 딕셔너리}, 분석타겟컬럼의 값  ) , ..... ]
# 분석 타겟 컬럼 : 라벨(label)
# 분석에 사용하는 데이터가 되는 컬럼 : 어트리뷰트(attribute) - 속성
# 분석에 사용하는 데이터의 값 : inputs

inputs = []          # 최종적으로 사용할 데이터셋의 형태가 리스트여야 하기 때문에 빈 리스트를 생성합니다.

import csv
file=open(r"c:\python\data\skin2.csv", "r")          # csv 파일로 데이터셋을 불러옴
fatliver=csv.reader(file)
inputss=[]
for i in fatliver:
    inputss.append(i)        # 데이터 값

labelss = ['gender', 'age', 'job', 'marry', 'car','coupon_react']        # 데이터의 라벨(컬럼명)

for data in inputss:        # 위처럼 리스트로 된 데이터값과 리스트로된 라벨(컬럼명)을 분석에 맞는 데이터형태로 바꾸는 과정.
    temp_dict = {}          # 데이터셋 = [ ( {데이터가 되는 컬럼의 키와 값으로 구성된 딕셔너리}, 분석타겟컬럼의 값  ) , ..... ] 의 형태로 되어있어야 분석할 수 있다.
    c=len(labelss)-1        # 데이터셋의 최종값을 타겟변수로 두었기 때문에 타겟변수는 데이터값 딕셔너리에 넣지 않습니다. 분석타겟변수의 위치를 잡아주는 값
    for i in range(c):      # 타겟변수를 제외한 나머지 변수들로 딕셔너리에 데이터를 입력
        if i != c:          # 생성한 딕셔너리와 넣지 않은 타겟변수를 분석을 위한 큰 튜플안에 입력
            temp_dict[labelss[i]] = data[i]
    inputs.append(tuple((temp_dict, True if data[c] == 'YES' else False)))          #



def partition_by(inputs, attribute):        # attribute에 따라 inputs(데이터)를 파티션하는 함수
    groups = defaultdict(list)
    for input in inputs:
        key = input[0][attribute]           # 특정 attribute의 값을 불러오고 해당 attribute의 input값을 list에 추가한다.
        groups[key].append(input)
    return groups


def partition_entropy_by(inputs, attribute):        # 위에서 attribute에 따라 inputs를 파티션한 파티션의 엔트로피를 계산하는 함수
    partitions = partition_by(inputs, attribute)
    return partition_entropy(partitions.values())

for key in ['gender', 'age', 'job', 'marry', 'car'] :
    #print( partition_by(inputs,key).values())
    print (key,partition_entropy( partition_by(inputs,key).values()))

#  의사결정나무 tree 로 주어진 입력값 input 를 분류하자  !
