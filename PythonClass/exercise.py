
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
file=open(r"c:\python\data\fatliver.csv", "r")          # csv 파일로 데이터셋을 불러옴
fatliver=csv.reader(file)
inputss=[]
for i in fatliver:
    inputss.append(i)        # 데이터 값

labelss = ['age','gender','drink','smoke','Fatliver']        # 데이터의 라벨(컬럼명)

for data in inputss:        # 위처럼 리스트로 된 데이터값과 리스트로된 라벨(컬럼명)을 분석에 맞는 데이터형태로 바꾸는 과정.
    temp_dict = {}          # 데이터셋 = [ ( {데이터가 되는 컬럼의 키와 값으로 구성된 딕셔너리}, 분석타겟컬럼의 값  ) , ..... ] 의 형태로 되어있어야 분석할 수 있다.
    c=len(labelss)-1        # 데이터셋의 최종값을 타겟변수로 두었기 때문에 타겟변수는 데이터값 딕셔너리에 넣지 않습니다. 분석타겟변수의 위치를 잡아주는 값
    for i in range(c):      # 타겟변수를 제외한 나머지 변수들로 딕셔너리에 데이터를 입력
        if i != c:          # 생성한 딕셔너리와 넣지 않은 타겟변수를 분석을 위한 큰 튜플안에 입력
            temp_dict[labelss[i]] = data[i]
    inputs.append(tuple((temp_dict, True if data[c] == 'yes' else False)))          #


def partition_by(inputs, attribute):        # attribute에 따라 inputs(데이터)를 파티션하는 함수
    groups = defaultdict(list)
    for input in inputs:
        key = input[0][attribute]           # 특정 attribute의 값을 불러오고 해당 attribute의 input값을 list에 추가한다.
        groups[key].append(input)
    return groups


def partition_entropy_by(inputs, attribute):        # 위에서 attribute에 따라 inputs를 파티션한 파티션의 엔트로피를 계산하는 함수
    partitions = partition_by(inputs, attribute)
    return partition_entropy(partitions.values())

for key in ['age','gender','drink','smoke'] :
    #print( partition_by(inputs,key).values())
    print (key,partition_entropy( partition_by(inputs,key).values()))

#  의사결정나무 tree 로 주어진 입력값 input 를 분류하자  !


def classify(tree, input):   # 질문을 받아서 tree 와 비교해서 쿠폰반응이 True 인지 False 인지 출력하는 함수
    # 잎 노드이면 값을 바로 반환하라 !
    if tree in [True, False]:
        return tree

   # 그게 아니라면 데이터의 변수로 파티션을 나누자
   # 키로 변수 값, 값으로는 서브트리를 나타내는 dict 를 사용하면 된다.
    attribute, subtree_dict = tree
    # print(type(attribute))  # <class 'str'>
    # print(type(subtree_dict)) # <class 'dict'>
    print(attribute)  # marry, age
    print(subtree_dict) # {'YES': ('age', {'30대': ('job', {'NO': ('gender', {'남': False, '여':..
                        # marry 와 age 의 결정트리에 해당하는 부분 다 출력

    #subtree_key2 = input.get('age')
    # print(subtree_key2)  40대
    subtree_key = input.get(attribute)
    #print(subtree_key) # YES, 40대, True

    if subtree_key not in subtree_dict:           # 키에 해당하는 서브트리가 존재하지 않으면
        subtree = subtree_dict[subtree_key]       # None 서브트리를 사용
    #print(subtree_dict) # {'YES': ('age', {'30대': ('job', {'NO': ('gender', {'남': False, '여': ('car', {'NO': False, 'YES': True, None: False}), None: False}), 'YES': True, None: True}), '20대':..
    #print(subtree_key)  # YES
    subtree = subtree_dict[subtree_key]           # 서브트리를 선택
    return classify(subtree, input)               # 이 과정을 재귀를 통해 잎 노드가 반환될 때까지 계속 수행


def build_tree_id3(inputs, split_candidates=None):  # 쿠폰반응 데이터를 가지고 결정트리를 만드는 함수

    # 파티션이 첫 단계면 입력된 데이터의 모든 변수를 파티션 기준 후보로 설정
    if split_candidates is None:
        split_candidates = inputs[0][0].keys()
    #print(inputs)
    #print(inputs[0])
    #print(inputs[0][0])
    #print(inputs[0][0].keys()) #(['gender', 'age', 'job', 'marry', 'car'])

    # 입력된 데이터에서 True 와 False 의 갯수를 세어본다.
    #print(inputs) # [({'gender': '여', 'age': '30대', 'job': 'NO', 'marry': 'YES', 'car': 'NO'}, False)]
    num_inputs = len(inputs)
    #print(num_inputs) # 1
    num_trues = len([label for item, label in inputs if label])
    #print(num_trues) # 0
    num_falses = num_inputs - num_trues # 1
    #print(num_falses)

    if num_trues == 0:
        return False                              # true가 없다면 false 잎 노드를 반환
    if num_falses == 0:
        return True                               # false가 없다면 true 잎 노드를 반환

    if not split_candidates:
        return num_trues >= num_falses            # 만약 사용할 변수가 없으면 많은 수를 반환

    best_attribute = min(split_candidates, key=partial(partition_entropy_by, inputs))     # 가장 적합한 변수(attribute)를 기준으로 파티션 시작
    partitions = partition_by(inputs, best_attribute)
    new_candidates = [a for a in split_candidates
                      if a != best_attribute]
    subtrees = {attribute_value: build_tree_id3(subset, new_candidates) for attribute_value, subset in      # 재귀적으로 돌아가면서 서브 트리를 구축
                partitions.items()}
    subtrees[None] = num_trues > num_falses  # 기본값
    return (best_attribute, subtrees)


tree = build_tree_id3(inputs)
# print(tree)

#['gender', 'age', 'job', 'marry', 'car']
answer = classify(tree, {"gender": '여자', "age": '40대', "job": 'YES', "marry": 'YES', "car" : "NO"})
#answer = classify(tree, {"gender": '여자', "age": '40대',"marry" :'YES'})
#answer = classify(tree, {"marry" :'YES'})
print(answer)



###################################################################################
"""
Created on Tue May  2 13:49:00 2017
@author: stu
"""
import matplotlib.pyplot as plt

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction', va="center", ha="center",
                            bbox=nodeType, arrowprops=arrow_args)


def createPlot():
    fig = plt.figure(1, facecolor="white")
    fig.clf()
    createPlot.ax1 = plt.subplot(111, frameon=False)
    plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()


def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth


def retrieveTree(i):
    listOfTrees = [
        {
            'no surfacing': {
                0: 'no',
                1: {
                    'flippers': {
                        0: 'no',
                        1: 'yes'
                    }
                }
            }
        },
        {
            'no surfacing': {
                0: 'no',
                1: {
                    'flippers': {
                        0: {
                            'head': {
                                0: 'no',
                                1: 'yes'
                            }
                        },
                        1: 'no'
                    }
                }
            }
        }]
    return listOfTrees[i]


def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)


def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW,
              plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff),
                     cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW;
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()


myTree = {'MARRY_YN': {'YES': {'AGE': {'40': 'YES', '30': {'JOB_YN': {'YES': 'YES', 'NO':
    {'GENDER': {'M': 'NO', 'F': {'CAR_YN': {'YES': 'YES', 'NO': 'NO'}}}}}}, '20': {'JOB_YN':
                                                                                       {'YES': 'NO', 'NO': {
                                                                                           'CAR_YN': {'YES': 'YES',
                                                                                                      'NO': 'NO'}}}}}},
                       'NO': 'NO'}}

print(createPlot(myTree))
###############################################################################


# ('gender',
#           {'남자': ('age',
#                       {'40대': ('smoke',
#                                       {'금연': ('drink',
#                                                    {'음주많음': False,
#                                                     '음주적음': False,
#                                                      None: False}),
#                                       '흡연': ('drink',
#                                                     {'음주많음': False,
#                                                     '음주적음': False,
#                                                     None: False}),
#                                        None: False}),
#                       '30대': ('smoke',
#                                       {'흡연': ('drink',
#                                                      {'음주많음': False,
#                                                      '음주적음': True,
#                                                       None: False}),
#                                         '금연': ('drink',
#                                                       {'음주적음': False,
#                                                       '음주많음': False,
#                                                        None: False}),
#                                         None: False}),
#                       '50대': ('smoke',
#                                        {'흡연': ('drink',
#                                                       {'음주많음': False,
#                                                       '음주적음': False,
#                                                        None: False}),
#                                         '금연': ('drink',
#                                                       {'음주적음': False,
#                                                        '음주많음': False,
#                                                         None: False}),
#                                          None: False}),
#                       None: False}),
#           '여자': ('age', {'50대': ('drink', {'음주적음': ('smoke', {'금연': False, '흡연': False, None: False}), '음주많음': ('smoke', {'흡연': True, '금연': True, None: True}), None: False}), '40대': ('smoke', {'금연': ('drink', {'음주적음': False, '음주많음': False, None: False}), '흡연': True, None: False}), '30대': ('smoke', {'금연': ('drink', {'음주적음': False, '음주많음': False, None: False}), '흡연': False, None: False}), None: False}), None: False})

