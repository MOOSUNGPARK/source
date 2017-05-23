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
                group.append(input[0][key])
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


