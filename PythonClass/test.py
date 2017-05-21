from librosa import load, stft, feature, display, get_duration
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics.pairwise as sklearn
from sklearn.metrics.pairwise import cosine_similarity


music = 'palette'
y, sr = load(r'c:\python\data\music\{}.mp3'.format(music), sr=882)
s = np.abs(stft(y)**2)
time = get_duration(y=y, sr=sr)

chroma = feature.chroma_stft(S=s, sr=sr)
chromaT=np.transpose(chroma,axes=(1,0))
cs= cosine_similarity(chromaT)

result = []
resultdic = {}
nodes = []
short = []
temp = []
for i in range(5):
    temp.append('cs[m+{}][n+{}]'.format(4*i,4*i))
ifcondition = ' + '.join(temp)

for m in range(len(cs)):
    try:
        for n in range(m-1):
            # if cs[m][n]>=0.98:
            if [m,n] not in short and eval(ifcondition)/5 >= 0.9:
                result.append( (int((time / len(cs)) * m),int((time / len(cs)) * n)))
                nodes.append((int((time / len(cs)) * m)))
                nodes.append((int((time / len(cs)) * n)))
                [short.append([m + i, n + i]) for i in range(20)]
                # result.append((int((time/len(cs))*n),int(time/len(cs)*m)))
                # # if int((time / len(cs)) * n) in resultdic:
                # #     resultdic[int((time / len(cs)) * n)] += 1
                # # else:
                # #     resultdic[int((time / len(cs)) * n)] = 1
                # # if int((time / len(cs)) * m) in resultdic:
                # #     resultdic[int((time / len(cs)) * m)] += 1
                # # else:
                # #     resultdic[int((time / len(cs)) * m)] = 1
                # resultdic[int((time/len(cs))*n)] = 1 if int((time/len(cs)) *n) not in resultdic else resultdic[int((time/len(cs))*n)] + 1
                # resultdic[int((time/len(cs))*m)] = 1 if int((time/len(cs)) *m) not in resultdic else resultdic[int((time / len(cs)) * m)] + 1
                # [short.append([m + i, n + i]) for i in range(20)]
    except IndexError:
        continue

# result.sort(key= lambda r : r[0])
# print(result)
# print(resultdic.keys())

#############################################

import networkx as nx


G = nx.MultiGraph()
G.add_nodes_from(list(set(nodes)))
G.add_edges_from(result)
# plt.show()

############ betweenness_centrality ###################
# centrality = nx.betweenness_centrality(G)
# highlight = max(centrality,key=centrality.get) - 0.8
# print(centrality)
# print(highlight)

############ eigenvector_centrality ###################
centrality = nx.eigenvector_centrality_numpy(G)
highlight = max(centrality,key=centrality.get) - 0.8
print(centrality)
print(highlight)


#############################################
from pygame import mixer
import pygame
########################
from ctypes import windll
SetWindowPos = windll.user32.SetWindowPos
#
# NOSIZE = 1
# NOMOVE = 2
# TOPMOST = -1
# NOT_TOPMOST = -2
#
# def alwaysOnTop(yesOrNo):
#     zorder = (NOT_TOPMOST, TOPMOST)[yesOrNo] # choose a flag according to bool
#     hwnd = pygame.display.get_wm_info()['window'] # handle to the window
#     SetWindowPos(hwnd, zorder, 0, 0, 0, 0, NOMOVE|NOSIZE)

#########################

pygame.init()
window = pygame.display.set_mode((100,100))


SetWindowPos(pygame.display.get_wm_info()['window'], -1, 0, 0, 0, 0, 0x0003)


mixer.init()
mixer.music.load(r'c:\python\data\music\{}.mp3'.format(music))
mixer.music.play(start=highlight)

pygame.time.wait(10000)
pygame.quit()


while True :
    mixer.music.fadeout( (highlight + 20) *1000)
    # mixer.music.stop()
    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            pygame.quit()

# circle = pygame.draw.circle(window, (50,30,90), (90,30),16,5)


# import timeit
# start = timeit.default_timer()
# while timeit.default_timer() - start <= 10:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             pygame.quit()
#     pygame.display.update()

#
# plt.figure(figsize=(10, 10))
# display.specshow(cs, y_axis ='time', x_axis='time')
# plt.colorbar()
# plt.title('{}'.format(music))
# plt.tight_layout()
# plt.show()



'''
import networkx as nx


G = nx.MultiGraph()
G.add_nodes_from([47, 122])
G.add_edges_from([(47, 122)])
# nx.draw(G)
# plt.show()

print(max(nx.betweenness_centrality(G),key=nx.betweenness_centrality(G).get))
print(nx.betweenness_centrality(G))
'''

