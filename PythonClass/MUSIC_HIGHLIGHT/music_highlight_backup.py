from librosa import load, stft, feature, get_duration
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import networkx as nx
from pygame import mixer,init, display, time, quit
from ctypes import windll

############### 플레이리스트 ###############
# bigbang / iluvit / inception / knock /
# palette / raindrop / rookie / russian /
# withcoffee / imissyou / ahyeah / newface /
# skyrim / noreply / soran / friday / wouldu /
# whistle / beautiful / loser / soso /

############### 플레이 정보 ###############
music_name = 'ahyeah'           # 노래 제목
play_duration = 20              # 재생 시간

#########################################
class Song(object):
    def __init__(self,music_name,show=False):
        self.music = music_name
        self.time = 0
        self.show = show
        self.result = []
        self.nodes = []
        self.alreadyexists = []
        ###################
        self.maxnode = 0
        ###################

    def LoadSong(self):
        y, sr = load(r'd:\python\data\music\{}.mp3'.format(self.music), sr=882)
        s = np.abs(stft(y)**2)
        self.time = get_duration(y=y, sr=sr)
        chroma = feature.chroma_stft(S=s, sr=sr)
        chromaT=np.transpose(chroma,axes=(1,0))
        print('Loading Finished(1/3)')
        return cosine_similarity(chromaT)

    def IfCondition(self):
        temp = []
        for i in range(10):
            temp.append('cs[m+{}][n+{}]'.format(2*i,2*i))
        return ' + '.join(temp)
##############################################################
    # def IfCondition2(self):
    #     temp = []
    #     for i in range(6):
    #         temp.append('cs[m+{}][n+{}]'.format(5*i,5*i))
    #     return ' + '.join(temp)
##############################################################
    def FindNodes(self, cs, converttime, ifcondition, accuracy):
        for m in range(len(cs)):
            try:
                for n in range(m-1):
                    if [m,n] not in self.alreadyexists and eval(ifcondition)/10 >= accuracy:
                        self.result.append((int(converttime * m),int(converttime * n)))
                        self.nodes.append((int(converttime * m)))
                        self.nodes.append((int(converttime * n)))
                        [self.alreadyexists.append([m + i, n + i]) for i in range(20)]
            except IndexError:
                continue

    def fibo(self, num):
        if num == 2:
            return 2
        elif num == 1 :
            return 1
        return self.fibo(num - 1) + self.fibo(num - 2)

#############################################################
    # def MaxNodes(self):
    #     cs = self.LoadSong()
    #     converttime = (self.time / len(cs))
    #     ifcondition = self.IfCondition2()
    #     maxvalue = 0
    #     for m in range(len(cs)):
    #         try:
    #             for n in range(m - 1):
    #                 if [m, n] not in self.alreadyexists and eval(ifcondition) >= maxvalue:
    #                     maxvalue = eval(ifcondition)
    #                     self.maxnode = [(int(converttime * m)),(int(converttime * n))]
    #                     [self.alreadyexists.append([m + i, n + i]) for i in range(10)]
    #         except IndexError:
    #             continue
    #     print('Making Nodes Finished(2/3)')
    #     return self.maxnode[1] - 0.5
#############################################################

    def MakeNodes(self):
        cs = self.LoadSong()
        converttime = (self.time / len(cs))
        ifcondition = self.IfCondition()
        trycnt = 0

        self.FindNodes(cs, converttime, ifcondition, accuracy=0.995)

        while len(self.result) <= 1 :
            trycnt += 1
            self.FindNodes(cs, converttime, ifcondition, accuracy=0.997 - 0.006 * self.fibo(trycnt))
            print('Changing Accuracy...')

        print('Making Nodes Finished(2/3)')
        return cs

    def Analysis(self):
        cs = self.MakeNodes()

        if self.show == True:
            plt.figure(figsize=(10, 10))
            display.specshow(cs, y_axis ='time', x_axis='time')
            plt.colorbar()
            plt.title('{}'.format(music_name))
            plt.tight_layout()
            plt.show()

        G = nx.MultiGraph()
        G.add_nodes_from(list(set(self.nodes)))
        G.add_edges_from(self.result)

        ########## eigenvector_centrality ##########
        centrality = nx.eigenvector_centrality_numpy(G)
        print('Analyzing Finished(3/3)')
        return max(centrality, key=centrality.get) - 0.5

        ########## betweenness_centrality ##########
        # centrality = nx.betweenness_centrality(G)
        # highlight = max(centrality,key=centrality.get) - 0.8


class Play(object):
    @staticmethod
    def PlaySong():
        song = Song(music_name)
        highlight = song.Analysis()
        ################################
        # highlight = song.MaxNodes()
        ################################
        init()
        mixer.init()
        display.set_mode((100,100))
        SetWindowPos = windll.user32.SetWindowPos
        SetWindowPos(display.get_wm_info()['window'], -1, 0, 0, 0, 0, 0x0003)
        mixer.music.load(r'd:\python\data\music\{}.mp3'.format(music_name))
        print('Music Start!')
        mixer.music.play(start=highlight)
        time.wait(play_duration * 1000)
        quit()

if __name__ == '__main__':
    Play.PlaySong()

########################################################################################
# import librosa
from librosa import load, stft, feature, display, get_duration
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics.pairwise as sklearn
import heapq
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

y, sr = load(r'd:\python\data\soso.mp3', sr=882)
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
                if int((time / len(cs)) * n) in resultdic:
                    resultdic[int((time / len(cs)) * n)] += 1
                else:
                    resultdic[int((time / len(cs)) * n)] = 1
                if int((time / len(cs)) * m) in resultdic:
                    resultdic[int((time / len(cs)) * m)] += 1
                else:
                    resultdic[int((time / len(cs)) * m)] = 1
                resultdic[int((time/len(cs))*n)] = 1 if resultdic[int((time/len(cs))*n)] == None else resultdic[int((time/len(cs))*n)] + 1
                resultdic[int((time/len(cs))*m)] = 1 if resultdic[int((time/len(cs))*m)] == None else resultdic[int((time / len(cs)) * m)] + 1
                [short.append([m + i, n + i]) for i in range(20)]
    except IndexError:
        continue

result.sort(key= lambda r : r[0])
print(result)
print(resultdic)


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

####################################################################################
from librosa import load, stft, feature, display, get_duration
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics.pairwise as sklearn
from sklearn.metrics.pairwise import cosine_similarity


music = 'soran'
y, sr = load(r'd:\python\data\music\{}.mp3'.format(music), sr=882)
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
for i in range(10):
    temp.append('cs[m+{}][n+{}]'.format(2*i,2*i))
ifcondition = ' >=0.9 and '.join(temp)

for m in range(len(cs)):
    try:
        for n in range(m-1):
            # if cs[m][n]>=0.98:
            if [m,n] not in short and eval(ifcondition) >= 0.9:
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
mixer.music.load(r'd:\python\data\music\{}.mp3'.format(music))
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














