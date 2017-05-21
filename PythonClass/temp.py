from librosa import load, stft, feature, display, get_duration
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics.pairwise as sklearn
from sklearn.metrics.pairwise import cosine_similarity


music = 'soso'
y, sr = load(r'd:\python\data\{}.mp3'.format(music), sr=882)
s = np.abs(stft(y)**2)
time = get_duration(y=y, sr=sr)

chroma = feature.chroma_stft(S=s, sr=sr)
D = stft(y)
chromaT=np.transpose(chroma,axes=(1,0))
cs= cosine_similarity(chromaT)

result = []
resultdic = {}
short = []
temp = []
for i in range(20):
    temp.append('cs[m+{}][n+{}]'.format(i,i))
ifcondition = ' >=0.9 and '.join(temp)

for m in range(len(cs)):
    try:
        for n in range(m-1):
            if [m,n] not in short and [m+1,n] not in short and [m,n+1] not in short and eval(ifcondition) >= 0.9:
                result.append((int((time/len(cs))*n),int(time/len(cs)*m)))
                # if int((time / len(cs)) * n) in resultdic:
                #     resultdic[int((time / len(cs)) * n)] += 1
                # else:
                #     resultdic[int((time / len(cs)) * n)] = 1
                # if int((time / len(cs)) * m) in resultdic:
                #     resultdic[int((time / len(cs)) * m)] += 1
                # else:
                #     resultdic[int((time / len(cs)) * m)] = 1
                resultdic[int((time/len(cs))*n)] = 1 if int((time/len(cs)) *n) not in resultdic else resultdic[int((time/len(cs))*n)] + 1
                resultdic[int((time/len(cs))*m)] = 1 if int((time/len(cs)) *m) not in resultdic else resultdic[int((time / len(cs)) * m)] + 1
                [short.append([m + i, n + i]) for i in range(20)]
    except IndexError:
        continue

result.sort(key= lambda r : r[0])
print(result)
print(resultdic.keys())





plt.figure(figsize=(10, 10))
display.specshow(cs, y_axis ='time', x_axis='time')
plt.colorbar()
plt.title('{}'.format(music))
plt.tight_layout()
plt.show()







