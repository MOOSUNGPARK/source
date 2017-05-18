# import librosa
from librosa import load, stft, feature, display
import numpy as np
import matplotlib.pyplot as plt
#
# print(chroma)
#
# D = librosa.stft(y)
# print(D)

y, sr = load('d:\python\data\with coffee.wma')
s = np.abs(stft(y)**2)
chroma = np.around(feature.chroma_stft(S=s, sr=sr, norm=None )*10**7,decimals=2, out=None)
D = np.around(stft(y),decimals=2,out=None)
print('chroma',chroma)

plt.figure(figsize=(10, 4))
display.specshow(chroma, y_axis ='chroma', x_axis='time')
plt.colorbar()
plt.title('Chromagram')
plt.tight_layout()
plt.show()