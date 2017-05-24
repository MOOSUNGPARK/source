import collections
import numpy as np
from collections import defaultdict

def class_probabilities(labels):
    values = collections.Counter(labels).values()
    return [value/sum(values) for value in values]

def entropy(labels):
    prob_list = class_probabilities(labels)
    return float(sum(-prob * np.log2(prob) for prob in prob_list if prob != 0))

def entropy_group(inputs):
    keys = inputs[0][0].keys()

    for key in keys :
        if key != 'cust_name':
            group = []
            for input in inputs:
                group.append(input[1])
            print(key, entropy(group))

def column_data(inputs, column):
    groups = defaultdict(list)
    for input in inputs:
        key = input[0][column]
        groups[key].append(input[1])
    return groups

def column_data_to_list(groups):
    result = []
    keys = groups.keys()
    for key in keys:
        result.append(groups[key])
    return result

def partition_entropy(groups):
    subsets = column_data_to_list(groups)
    total_count = sum(len(subset) for subset in subsets)
    return sum(entropy(subset) * len(subset)/total_count for subset in subsets)

if __name__ == '__main__':
    inputs = [
        ({'cust_name': 'SCOTT', 'card_yn': 'Y', 'review_yn': 'Y', 'before_buy_yn': 'Y'}, True),
        ({'cust_name': 'SMITH', 'card_yn': 'Y', 'review_yn': 'Y', 'before_buy_yn': 'Y'}, True),
        ({'cust_name': 'ALLEN', 'card_yn': 'N', 'review_yn': 'N', 'before_buy_yn': 'Y'}, False),
        ({'cust_name': 'JONES', 'card_yn': 'Y', 'review_yn': 'N', 'before_buy_yn': 'N'}, True),
        ({'cust_name': 'WARD', 'card_yn': 'Y', 'review_yn': 'Y', 'before_buy_yn': 'Y'}, True)]

    entropy_group(inputs)

    for column in inputs[0][0].keys():
        if column != 'cust_name':
            print(column ,partition_entropy(column_data(inputs, column)))




'''

groups defaultdict(<class 'list'>, {'Y': [True, True, True, True], 'N': [False]})
result [[True, True, True, True], [False]]
5
dict_values([4])
dict_values([1])
card_yn 0.0
groups defaultdict(<class 'list'>, {'Y': [True, True, True], 'N': [False, True]})
result [[True, True, True], [False, True]]
5
dict_values([3])
dict_values([1, 1])
review_yn 0.4
groups defaultdict(<class 'list'>, {'Y': [True, True, False, True], 'N': [True]})
result [[True, True, False, True], [True]]
5
dict_values([3, 1])
dict_values([1])
before_buy_yn 0.6490224995673063

'''



inputs = []  # 최종적으로 사용할 데이터셋의 형태가 리스트여야 하기 때문에 빈 리스트를 생성합니다.

import csv

file = open(r"c:\python\data\skin2.csv", "r")  # csv 파일로 데이터셋을 불러옴
fatliver = csv.reader(file)
inputss = []
for i in fatliver:
    inputss.append(i)  # 데이터 값

labelss = ['gender', 'age', 'job', 'marry', 'car', 'coupon_react']  # 데이터의 라벨(컬럼명)

for data in inputss:  # 위처럼 리스트로 된 데이터값과 리스트로된 라벨(컬럼명)을 분석에 맞는 데이터형태로 바꾸는 과정.
    temp_dict = {}  # 데이터셋 = [ ( {데이터가 되는 컬럼의 키와 값으로 구성된 딕셔너리}, 분석타겟컬럼의 값  ) , ..... ] 의 형태로 되어있어야 분석할 수 있다.
    c = len(labelss) - 1  # 데이터셋의 최종값을 타겟변수로 두었기 때문에 타겟변수는 데이터값 딕셔너리에 넣지 않습니다. 분석타겟변수의 위치를 잡아주는 값
    for i in range(c):  # 타겟변수를 제외한 나머지 변수들로 딕셔너리에 데이터를 입력
        if i != c:  # 생성한 딕셔너리와 넣지 않은 타겟변수를 분석을 위한 큰 튜플안에 입력
            temp_dict[labelss[i]] = data[i]
    inputs.append(tuple((temp_dict, True if data[c] == 'YES' else False)))  #