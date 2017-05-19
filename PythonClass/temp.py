# import librosa
from librosa import load, stft, feature, display
import numpy as np
import matplotlib.pyplot as plt
#
# print(chroma)
#
# D = librosa.stft(y)
# print(D)
import math
def cosine_similarity(v1,v2):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy/math.sqrt(sumxx*sumyy)

y, sr = load('c:\python\data\Super Fantastic.mp3')
s = np.abs(stft(y)**2)
# chroma1 = np.around(feature.chroma_stft(S=s, sr=sr, norm=None )*10**7,decimals=2, out=None)
chroma = np.around(feature.chroma_stft(S=s, sr=sr) * 10, decimals=2, out=None)
D = stft(y)
print(cosine_similarity(chroma[0],chroma[3]))
print(len(chroma[0]))
print(len(chroma[1]))
print(len(chroma[2]))
print(len(chroma))
print(chroma)

plt.figure(figsize=(10, 4))
display.specshow(chroma, y_axis ='chroma', x_axis='time')
plt.colorbar()
plt.title('Chromagram')
plt.tight_layout()
plt.show()