# import librosa
from librosa import load, stft, feature, display, get_duration
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics.pairwise as sklearn

#
# print(chroma)
#
# D = librosa.stft(y)
# print(D)
# import math
# def cosine_similarity(v1,v2):
#     "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
#     sumxx, sumxy, sumyy = 0, 0, 0
#     for i in range(len(v2)):
#         x = v1[i]; y = v2[i]
#         sumxx += x*x
#         sumyy += y*y
#         sumxy += x*y
#     return sumxy/math.sqrt(sumxx*sumyy)
from sklearn.metrics.pairwise import cosine_similarity

y, sr = load(r'c:\python\data\knock.mp3', sr=882)
s = np.abs(stft(y)**2)
time = get_duration(y=y, sr=sr)
# chroma1 = np.around(feature.chroma_stft(S=s, sr=sr, norm=None )*10**7,decimals=2, out=None)
# chroma = np.around(feature.chroma_stft(S=s, sr=sr) * 10, decimals=2, out=None

chroma = feature.chroma_stft(S=s, sr=sr)
D = stft(y)
chromaT=np.transpose(chroma,axes=(1,0))
cs= sklearn.cosine_similarity(chromaT)
# print(chromaT)

# for i in range(len(chroma[0])):
    # print(cosine_similarity(chroma[:,i], chromaT))

# print(cosine_similarity((chromaT[0], chromaT)))
# print(chromaT[0])
# print(chromaT)
# print(cosine_similarity(chromaT[0],chroma[:,0]))

# print(len(cs))
# print(len(cs[0]))
# print(cs)
# print(cs[0][0])
# print(cs[100][0])
# print(cs[100][100])

result = []
short = []
temp = []
for i in range(20):
    temp.append('cs[m+{}][n+{}]'.format(i,i))
ifcondition = ' >=0.85 and '.join(temp)

for m in range(len(cs)):
    try:
        for n in range(m-1):
            if [m,n] not in short and [m+1,n] not in short and [m,n+1] not in short and eval(ifcondition) >= 0.9:
                result.append((int((time/len(cs))*n),int(time/len(cs)*m)))
                [short.append([m + i, n + i]) for i in range(20)]
    except IndexError:
        continue

result.sort(key= lambda r : r[0])

print(len(result))
print(result)


# print(chroma[:,:])
# print(len(chroma))
# print(len(chroma[0]))
# final_list = []
# for i in range(len(chroma[0])):
#     chroma_list = []
#     for j in range(len(chroma[0])):
#         chroma_list.append(cosine_similarity(chroma[:,i], chroma[:,j]))
#     final_list.append(chroma_list)
# print(final_list)

'''
plt.figure(figsize=(10, 4))
display.specshow(chroma, y_axis ='chroma', x_axis='time')
plt.colorbar()
plt.title('Chromagram')
plt.tight_layout()
plt.show()
'''
plt.figure(figsize=(10, 10))
display.specshow(cs, y_axis ='time', x_axis='time')
plt.colorbar()
plt.title('Chromagram_two')
plt.tight_layout()
plt.show()







