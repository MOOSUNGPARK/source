import numpy as np

# 1. support vector machine

# 비용함수
def L_i_vectorized(x, y, W):
    scores = W.dot(x)
    margins = np.maximum(0, scores - scores[y] + 1)
    margins[y] = 0  # y 의 마진은 scores - scores[y] + 1 = 1 이 되므로 불필요하게 커짐. 이를 없애기 위해 0으로 바꿔줌
    loss_i = np.sum(margins)
    return loss_i

