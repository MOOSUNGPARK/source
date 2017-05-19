# import librosa
from librosa import load, stft, feature, display
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

y, sr = load('d:\python\data\knock.mp3', sr=4410)
s = np.abs(stft(y)**2)
# chroma1 = np.around(feature.chroma_stft(S=s, sr=sr, norm=None )*10**7,decimals=2, out=None)
# chroma = np.around(feature.chroma_stft(S=s, sr=sr) * 10, decimals=2, out=None
chroma = feature.chroma_stft(S=s, sr=sr)
D = stft(y)
# for i in range(12):
#     print(cosine_similarity((chroma[i], chroma)))
# print(cosine_similarity(chroma[0],chroma))
# print(cosine_similarity(chroma[1],chroma))
# print(cosine_similarity(chroma[2],chroma))
# print(cosine_similarity(chroma[3],chroma))
# print(cosine_similarity(chroma[4],chroma))
# print(cosine_similarity(chroma[5],chroma))
# print(cosine_similarity(chroma[6],chroma))
# print(cosine_similarity(chroma[7],chroma))
# print(cosine_similarity(chroma[8],chroma))
# print(cosine_similarity(chroma[9],chroma))
# print(cosine_similarity(chroma[10],chroma))
# print(cosine_similarity(chroma[11],chroma))
# print(chroma[:,0])
# print(chroma[:,1])
chromaT=np.transpose(chroma,axes=(1,0))
# print(chromaT)

# for i in range(len(chroma[0])):
    # print(cosine_similarity(chroma[:,i], chromaT))

# print(cosine_similarity((chromaT[0], chromaT)))
# print(chromaT[0])
# print(chromaT)
# print(cosine_similarity(chromaT[0],chroma[:,0]))
cs= sklearn.cosine_similarity(chromaT)
print(len(cs))
print(len(cs[0]))
print(cs)
# print(cs[0][0])
# print(cs[100][0])
# print(cs[100][100])

result = []

temp = []
for i in range(100):
    temp.append('cs[m+{}][n+{}]'.format(i,i))
ifcondition = ' >=0.9 and '.join(temp)

for m in range(len(cs)):
    for n in range(len(cs)):
        try:
            if m>n and eval(ifcondition) >= 0.9:
                result.append(((m,n),(m+99,n+99)))
        except IndexError:
            continue
print(len(result))



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
plt.figure(figsize=(10, 4))
display.specshow(cs, y_axis ='time', x_axis='time')
plt.colorbar()
plt.title('Chromagram_two')
plt.tight_layout()
plt.show()




'''
superfantastic
        0             24        48           71          95          119         143         167        191          214        238          262
        1             2           3           4           5           6           7           8          9           10          11           12
1 [[ 1.          0.87817053  0.78308425  0.71289377  0.70358819  0.72829723  0.70287041  0.71691167  0.73586437  0.75150505  0.80246725  0.90045949]]
2 [[ 0.87817053  1.          0.84024741  0.74347301  0.70487722  0.7359617   0.72697458  0.69065586  0.7194341   0.7025845   0.74434697  0.76181899]]
3 [[ 0.78308425  0.84024741  1.          0.88073394  0.80001328  0.71087735  0.67393635  0.70727247  0.67297438  0.68343322  0.67773995  0.71595698]]
4 [[ 0.71289377  0.74347301  0.88073394  1.          0.86951454  0.72072723  0.66935344  0.67334071  0.66785942  0.67574236  0.68548773  0.69376413]]
5 [[ 0.70358819  0.70487722  0.80001328  0.86951454  1.          0.85188324  0.71514334  0.71759275  0.67761335  0.69078212  0.68382013  0.68450202]]
6 [[ 0.72829723  0.7359617   0.71087735  0.72072723  0.85188324  1.          0.85466682  0.77837066  0.72903864  0.72106011  0.7447841   0.71275491]]
7 [[ 0.70287041  0.72697458  0.67393635  0.66935344  0.71514334  0.85466682  1.          0.87769695  0.73262536  0.70660183  0.71838179  0.70630271]]
8 [[ 0.71691167  0.69065586  0.70727247  0.67334071  0.71759275  0.77837066  0.87769695  1.          0.85588283  0.77156983  0.70833567  0.70908095]]
9 [[ 0.73586437  0.7194341   0.67297438  0.66785942  0.67761335  0.72903864  0.73262536  0.85588283  1.          0.90315311  0.76706711  0.74023852]]
10[[ 0.75150505  0.7025845   0.68343322  0.67574236  0.69078212  0.72106011  0.70660183  0.77156983  0.90315311  1.          0.87326556  0.78535715]]
11[[ 0.80246725  0.74434697  0.67773995  0.68548773  0.68382013  0.7447841   0.71838179  0.70833567  0.76706711  0.87326556  1.          0.88544465]]
12[[ 0.90045949  0.76181899  0.71595698  0.69376413  0.68450202  0.71275491  0.70630271  0.70908095  0.74023852  0.78535715  0.88544465  1.        ]] 
'''
'''
트와이스 녹녹
        1 0         2  22       3  44      4  1.27     5  1.49     6  2.11      7  2.33    8.  2.55    9   3.16    10.  3.38    11.  4                     
1 [[ 1.          0.87160648  0.83334613  0.77743579  0.70939762  0.69938781   0.66624598  0.74550016  0.76019356  0.77630943  0.83412     0.88634919]]
2 [[ 0.87160648  1.          0.88608878  0.76443254  0.69648503  0.6935761    0.6600369   0.69978807  0.69622433  0.70995325  0.74173788  0.73745178]]
3 [[ 0.83334613  0.88608878  1.          0.88891149  0.80017428  0.7836477    0.70028764  0.74233756  0.71305329  0.72592052  0.75599548  0.75767977]]
4 [[ 0.77743579  0.76443254  0.88891149  1.          0.86252708  0.81104083   0.72440389  0.76877038  0.759865    0.72221193  0.73419017  0.73665976]]
5 [[ 0.70939762  0.69648503  0.80017428  0.86252708  1.          0.90315425   0.71480317  0.72357397  0.70736173  0.70888631  0.69617229  0.7045137 ]]
6 [[ 0.69938781  0.6935761   0.7836477   0.81104083  0.90315425  1.           0.84514271  0.77421753  0.68917644  0.70293535  0.70504193  0.67231201]]
7 [[ 0.66624598  0.6600369   0.70028764  0.72440389  0.71480317  0.84514271   1.          0.85307412  0.69007773  0.70255204  0.71345956  0.65471718]]
8 [[ 0.74550016  0.69978807  0.74233756  0.76877038  0.72357397  0.77421753   0.85307412  1.          0.85664779  0.81401132  0.79294694  0.7165299 ]]
9 [[ 0.76019356  0.69622433  0.71305329  0.759865    0.70736173  0.68917644   0.69007773  0.85664779  1.          0.8580789   0.76344271  0.73077413]]
10[[ 0.77630943  0.70995325  0.72592052  0.72221193  0.70888631  0.70293535   0.70255204  0.81401132  0.8580789   1.          0.90051864  0.76757147]]
11[[ 0.83412     0.74173788  0.75599548  0.73419017  0.69617229  0.70504193   0.71345956  0.79294694  0.76344271  0.90051864  1.          0.87093424]]
12[[ 0.88634919  0.73745178  0.75767977  0.73665976  0.7045137   0.67231201   0.65471718  0.7165299   0.73077413  0.76757147  0.87093424  1.        ]]
'''




















