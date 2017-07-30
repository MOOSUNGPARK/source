import numpy as np

# 6x6 행렬 만들기
a = np.array([i for i in range(36)]).reshape(6,6)


# 3x3 필터 만들기
Filter = np.eye(3,3)

# 행렬 확인
print('---a\n',a)
print('---filter\n',Filter)

############################################################################

### 합성곱 연산 방법 1 ###

# 단일 곱셈-누산 vs 행렬곱 연산
d = np.array([[1,2,3],[4,5,6],[7,8,9]])
print('---d\n',d)
print('---단일 곱셈-누산 결과\n', np.sum(Filter * d)) # (1 * 1) + (-1 * 2) + (-1 * 3) + (1 * 4)
print('---행렬곱 연산 결과\n', np.dot(Filter, d))

# 넘파이 array indexing
print('---a[:,:]\n',a[:,:])            # a 전체 출력
print('---a[:,1:2]\n',a[:,0:3])      # a의 전체행 / 첫번째열~세번째열 출력
print('---a[0:3,4:5]\n',a[3:5,4:5])    # a의 네번재행~다섯번째행 / 다섯번째열 출력

# 스트라이드
for rn in range(len(a[0])-1):
    for cn in range(len(a[1])-1):
        print('---',[rn,cn],'\n',a[rn:rn+2, cn:cn+2])

# 합성곱 연산
result = []

for rn in range(len(a[0])-2):
    for cn in range(len(a[1])-2):
        result.append(np.sum(a[rn:rn+3, cn:cn+3] * Filter))

print('---result\n',result)
print('---len(result)\n', len(result))
len_a = int(np.sqrt(len(result)))
result = np.array(result).reshape(len_a,len_a)
print('---result.reshape\n', result)

# 패딩
a_pad = np.pad(a, pad_width=1, mode='constant', constant_values=0)
print('---a_pad\n',a_pad)

a_pad2 = np.pad(a, pad_width=2, mode='constant', constant_values=-1) # constant_values로 숫자 변경 가능
print('---a_pad2\n',a_pad2)

a_pad3 = np.pad(a, pad_width=((1,2),(3,4)), mode='constant', constant_values=0) # pad_width=( (위, 아래), (왼쪽, 오른쪽 패드 수) )
print('---a_pad3\n',a_pad3)

# 패딩 적용한 합성곱 연산
result2 = []

for rn in range(len(a_pad[0])-2):
    for cn in range(len(a_pad[1])-2):
        result2.append(np.sum(a_pad[rn:rn+3, cn:cn+3] * Filter))

print('---result2\n',result2)
print('---len(result2)\n', len(result2))
len_a2 = int(np.sqrt(len(result2)))
result2 = np.array(result2).reshape(len_a2,len_a2)
print('---result2.reshape\n', result2)


# 문제(1). 0부터 143까지 원소로 이뤄진 12x12 행렬을 만들고, 4x4 필터(단위 행렬)를 이용해 합성곱을 해보세요.
#         (단, 스트라이드는 1, 출력 행렬은 12x12가 되도록 패딩을 적용하세요)


############################################################################

### 합성곱 연산 2 ###

# 단일 곱셈 누산 -> 행렬곱 연산
print('---Filter\n', Filter)
print('---Filter.flatten()\n', Filter.flatten())  # flatten은 행렬을 벡터로 만들어줌

# 행렬곱하기 좋게 행렬을 변환해주는 함수
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

print('---변환 전 a\n', a)
print('---변환 후 a\n', im2col_sliding_strided(a, [3,3]))

# 행렬곱 연산을 이용한 합성곱

a_pad = np.pad(a, pad_width=1, mode='constant', constant_values=0)
a2 = im2col_sliding_strided(a_pad, [3,3])
Filter2 = Filter.flatten()
result = np.dot(a2, Filter2)
print('---합성곱 결과\n', result)
result = result.reshape(6,6)
print('---최종 결과\n', result)

# 문제(2). 앞에서 배운 두 가지 합성곱 방법을 각각 이용하여 0~1사이의 난수로 이루어진 300x300 행렬을
#          9x9 필터(단위행렬)를 이용해 합성곱을 해보세요. (단, 스트라이드는 1, 출력 행렬 크기는 300x300이 되도록 패딩을 적용하세요)





