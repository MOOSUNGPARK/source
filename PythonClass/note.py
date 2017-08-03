import numpy as np
import time

def padding(input, inputsize, outputsize, filtersize, stride=1):
    padsize = (stride * (outputsize - 1) - inputsize + filtersize)/2
    return np.pad(input, pad_width=(round(padsize+1e-7), int(padsize+1e-7)), mode='constant', constant_values=0)

def im2col_sliding_strided(A, filtersize, stepsize=1): # A = 변환할 행렬, filtersize = 필터 크기, stepsize = 스트라이드
    m, n = A.shape
    s0, s1 = A.strides
    BSZ = [m + 1 - filtersize[0], n + 1 - filtersize[1]]
    nrows = m - BSZ[0] + 1
    ncols = n - BSZ[1] + 1
    shp = BSZ[0], BSZ[1], nrows, ncols
    strd = s0, s1, s0, s1

    out_view = np.lib.stride_tricks.as_strided(A, shape=shp, strides=strd)
    return out_view.reshape(BSZ[0] * BSZ[1], -1)[:, ::stepsize]

# 0~1 사이의 난수 어레이 생성 + 패드 적용
# a = np.pad(np.random.rand(300,300), pad_width=4, mode='constant', constant_values=0)
a = padding(np.random.rand(300,300), 300, 300, 9)

# im2col
stime = time.time() # im2col 시작시간
a_converted = im2col_sliding_strided(a, (9,9), stepsize=1)
Filter = np.eye(9,9).flatten()
print('1) im2col_Filter\n', Filter)
result = np.dot(a_converted, Filter).reshape(300,-1)
print('1) im2col_result\n', result)
etime = time.time() # im2col 종료시간
print('1) im2col_수행시간\n', round(etime-stime, 6))

# 이중루프
stime2 = time.time() # 이중루프 시작시간
result2 = []
Filter2 = np.eye(9,9)
print('\n2) loop_Filter\n', Filter2)
for rn in range(len(a[0])-8):
    for cn in range(len(a[1])-8):
        result2.append(np.sum(a[rn:rn+9,cn:cn+9] * Filter2))
result2 = np.array(result2).reshape(300,300)
print('2) loop_result\n',result2)
etime2 = time.time() # 이중루프 종료시간
print('2) 이중루프_수행시간\n', round(etime2-stime2, 6))