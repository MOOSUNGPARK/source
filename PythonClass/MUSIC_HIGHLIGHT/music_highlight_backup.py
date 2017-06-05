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

from librosa import load, stft, feature, get_duration
import librosa.display as ld
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import networkx as nx
from pygame import mixer,init, display, time, quit
from ctypes import windll
from copy import deepcopy
from random import choice
import os

################ 플레이 정보 ################

play_duration = 10                # 재생 시간
file_loc = 'c:/python/data/music' # 노래 위치

###########################################

class Song(object):
    def __init__(self,show=False):
        self.music_dict = {}
        self.music = ''
        self.time = 0
        self.show = show
        self.result = []
########################################################################
    def LoadSong(self):
        self.music_dict = self.SearchSong(file_loc)
        self.music = self.Input()

        if self.music.upper() == 'RANDOM':
            self.music = choice(list(self.music_dict.keys()))

        y, sr = load(self.music_dict[self.music], sr=882)
        s = np.abs(stft(y)**2)
        self.time = get_duration(y=y, sr=sr)
        chroma = feature.chroma_stft(S=s, sr=sr)
        chromaT = np.transpose(chroma,axes=(1,0))
        print('\nLoading Finished!')
        return cosine_similarity(chromaT)

    def SearchSong(self, dirname):
        filenames = os.listdir(dirname)
        music_dict = {}
        for filename in filenames:
            full_filename = os.path.join(dirname, filename)
            ext = os.path.splitext(full_filename)[-1]
            file = os.path.splitext(filename)[0]
            if ext == '.mp3':
                music_dict[file] = full_filename
        return music_dict

    def Input(self):
        print('---------------------------------------------'
              'Music List---------------------------------------------')
        music_dict_list = list(self.music_dict.keys())
        for idx in range(len(music_dict_list)//5 + 1):
            try:
                print(' '.join([i.ljust(25) for i in music_dict_list[5 * idx : 5 * idx +5]]))
            except IndexError:
                print(' '.join([i.ljust(25) for i in music_dict_list[5 * idx: -1]]))

        return input('\n원하는 노래 제목을 입력하세요.(랜덤 원할 경우 random 입력) ')

########################################################################################
    def Analysis(self):
        chroma = self.MakeNodes()
        self.Chromagram(chroma, show=self.show)
        return self.result[0] - 1.5

    def MakeNodes(self):
        chroma = self.LoadSong()
        converttime = (self.time / len(chroma))
        filtered_chroma = self.Filtering(chroma)
        filterrate = 0.25
        while filtered_chroma.all() == 0 :
            filtered_chroma = self.Filtering(chroma, filterrate= filterrate-0.05)
        self.FindNodes(filtered_chroma, converttime)
        return filtered_chroma

    def Filtering(self, chroma, cnt=3, filterrate = 0.25):
        recursive_cnt = cnt
        chroma_copy = deepcopy(chroma)
        chroma = np.zeros((len(chroma), (len(chroma))))

        for rn in range(len(chroma) - 8):
            for cn in range(len(chroma) - 8):
                chroma[rn:rn + 9, cn:cn + 9] += self.Tensor(chroma_copy[rn:rn + 9, cn:cn + 9])
        chroma[chroma <= filterrate * np.max(chroma)] = 0
        # chroma[chroma <= 0] = 0

        if cnt == 0:
            # chroma[chroma <= 0.3 * np.max(chroma)] = 0
            return self.Normalization(chroma)

        print('Count down', recursive_cnt)
        return self.Filtering(chroma, cnt=recursive_cnt - 1)

    def FindNodes(self,chroma,converttime):
        for rn in range(len(chroma)):
            for cn in range(rn):
                chroma[rn][cn] = 0
        chroma[chroma <= 0] = 0
        frequency = self.LineFilter(chroma)
        best_frequency = round(converttime * max(frequency, key = lambda item: item[1])[0],1)
        print('\nMusic Name : {}'.format(self.music.upper()))
        print('Highlight : {}m {}s'.format(int(best_frequency//60), int(best_frequency%60)))
        self.result.append(best_frequency)

########################################################################################
    def LineFilter(self,chroma, maxcorrectcnt=3, line=25):
        maxcnt = maxcorrectcnt
        shorterline = line
        frequency = []

        for cn in range(len(chroma)-line):
            correctcnt = 0
            for rn in range(len(chroma)-line):
                cnt = 0
                while chroma[rn+cnt][cn+cnt] != 0 and cnt < line:
                    cnt += 1
                if cnt == line:
                    correctcnt += 1
            frequency.append([cn,correctcnt])

        if line <= 5 :
            return self.LineFilter(chroma, maxcorrectcnt=maxcnt-1, line=25)

        if max(frequency, key=lambda k:k[1])[1] >= maxcnt:
            # print(max(frequency, key=lambda k:k[1])[1])
            # print(frequency)
            return frequency

        return self.LineFilter(chroma, maxcorrectcnt=maxcnt, line=shorterline-5)

    def Tensor(self,chroma):
        tensor = np.zeros((9, 9))

        plus = (chroma[0][0] + chroma[1][1] + chroma[2][2] +
                chroma[3][3] + chroma[4][4] + chroma[5][5] +
                chroma[6][6] + chroma[7][7] + chroma[8][8])
        tensor[4][4] = ((10 / 9) * plus - (1 / 9) * np.sum(chroma)) / 9

        return tensor

########################################################################################
    def Normalization(self,chroma):
        for idx in range(len(chroma)):
            chroma[idx][idx] = 0
        return (chroma-np.mean(chroma))/np.max(chroma)

########################################################################################
    def Chromagram(self, chroma, show=False):
        if show == True:
            plt.figure(figsize=(10, 10))
            ld.specshow(chroma, y_axis='time', x_axis='time')
            plt.colorbar()
            plt.title('{}'.format(self.music))
            plt.tight_layout()
            plt.show()


    # def BestLine(self,chroma):
    #     longest=[]
    #     temp=0
    #     for cn in range(len(chroma)):
    #         for rn in range(cn):
    #             cnt=0
    #             temp=0
    #             while chroma[rn+cnt][cn+cnt] != 0:
    #                 cnt += 1
    #                 temp += 1
    #             # if temp >= 10:
    #             longest.append(temp)
    #             # if longest <= temp:
    #             #     longest = deepcopy(temp)
    #     linelist=list(set(longest))
    #     # linelist=longest
    #     print(sorted(linelist, reverse=True))
    #     # return sorted(longest, reverse=True)[1]
    #     print(sorted(linelist, reverse=True)[int(len(linelist) * 0.75)])
    #     # return sorted(linelist, reverse=True)[int(len(linelist) * 0.75)]
    #     return 20

class Play(object):
    @staticmethod
    def PlaySong():
        song = Song()
        highlight = song.Analysis()

        init()
        mixer.init()
        display.set_mode((100,100))
        SetWindowPos = windll.user32.SetWindowPos
        SetWindowPos(display.get_wm_info()['window'], -1, 0, 0, 0, 0, 0x0003)
        mixer.music.load(song.music_dict[song.music])
        print('Music Start!')
        mixer.music.play(start=highlight)
        # display.iconify()
        time.wait(play_duration * 1000)
        quit()

if __name__ == '__main__':
    Play.PlaySong()
    # song = Song(music_name)
    #
    # chroma = song.LoadSong()
    # chroma_copy = deepcopy(chroma)
    # tensor = np.array([[3,-1,-1],[-1,3,-1],[-1,-1,3]])
    # print('chroma',chroma[0:0+3,0:0+3])
    # print('tensor',chroma_copy[0:0+3,0:0+3] * tensor)
    # chroma[0:0 + 3, 0:0 + 3] += chroma_copy[0:0 + 3, 0:0 + 3] * tensor
    # print('tensored chroma', chroma)

    # print('행',len(chroma) -2 ,'열',len(chroma[0]) -2)
    # for rownum in range(len(chroma)-2):
    #     for colnum in range(len(chroma)-2):
    #             chroma[rownum:rownum+3, colnum:colnum+3] += chroma_copy[rownum:rownum+3, colnum:colnum+3] * tensor
    # for rownum in range(len(chroma)):
    #     for colnum in range(len(chroma)):
    #         int(chroma[rownum,colnum])
            # chroma[rownum,colnum] = 1 if chroma[rownum,colnum]>=1 else 0

    #
    #
    # def Tensor(chroma):
    #     tensor = np.zeros((3,3))
    #     plus = (chroma[0][0] + chroma[1][1] + chroma[2][2])
    #     minus = (chroma[0][1] + chroma[0][2] + chroma[1][0]+chroma[1][2] + chroma[2][0] + chroma[2][1])
    #     tensor[1][1] = (plus -0.5*minus)/3
    #     return tensor
    #
    #
    #
    # def Tensor_20(chroma):
    #     tensor = np.zeros((19, 19))
    #     plus = (chroma[0][0] + chroma[1][1] + chroma[2][2] + chroma[3][3] + chroma[4][4] + chroma[5][5] + chroma[6][6] +
    #             chroma[7][7] + chroma[8][8] + chroma[9][9] + chroma[10][10] + chroma[11][11] + chroma[12][12] + chroma[13][13]
    #             + chroma[14][14]+ chroma[15][15]+ chroma[15][15]+ chroma[16][16]+ chroma[17][17]+ chroma[18][18])
    #     tensor[9][9] = ((21 / 20) * plus - (1 / 20) * np.sum(chroma)) / 20
    #     return tensor
    #
    #
    # def Filtering(chroma,cnt=20):
    #     recursive_cnt = cnt
    #     # tensor = np.array([[3,-1,-1],[-1,3,-1],[-1,-1,3]])
    #     chroma_copy = deepcopy(chroma)
    #     chroma = np.zeros((len(chroma),(len(chroma))))
    #
    #     for rownum in range(len(chroma) - 2):
    #         for colnum in range(len(chroma) - 2):
    #             chroma[rownum:rownum + 3, colnum:colnum + 3] += Tensor(chroma_copy[rownum:rownum + 3, colnum:colnum + 3])
    #     for rownum in range(len(chroma)):
    #         for colnum in range(len(chroma)):
    #             chroma[rownum, colnum] = 0 if chroma[rownum, colnum] <= 0 else chroma[rownum, colnum]
    #     if cnt == 0 :
    #         return chroma
    #     print('cnt=',recursive_cnt,'chroma',chroma)
    #     return Filtering(chroma, cnt= recursive_cnt-1)
    #
    #
    # def Tensor_10(chroma):
    #     tensor = np.zeros((9,9))
    #     plus = (chroma[0][0] + chroma[1][1] + chroma[2][2] + chroma[3][3]+ chroma[4][4]+ chroma[5][5]+ chroma[6][6]+ chroma[7][7]+ chroma[8][8])
    #     tensor[4][4]=((10/9) * plus - (1/9) * np.sum(chroma))/9
    #     return tensor
    #
    # def Filtering_10(chroma,cnt=10):
    #     recursive_cnt = cnt
    #     # tensor = np.array([[3,-1,-1],[-1,3,-1],[-1,-1,3]])
    #     chroma_copy = deepcopy(chroma)
    #     chroma = np.zeros((len(chroma),(len(chroma))))
    #
    #     for rownum in range(len(chroma) - 8):
    #         for colnum in range(len(chroma) - 8):
    #             chroma[rownum:rownum + 9, colnum:colnum + 9] += Tensor_10(chroma_copy[rownum:rownum + 9, colnum:colnum + 9])
    #     for rownum in range(len(chroma)):
    #         for colnum in range(len(chroma)):
    #             chroma[rownum, colnum] = 0 if chroma[rownum, colnum] <= 0 else chroma[rownum, colnum]
    #     if cnt == 0 :
    #         return chroma
    #     print('cnt=',recursive_cnt,'chroma',chroma)
    #     return Filtering_10(chroma, cnt= recursive_cnt-1)
    #
    # def Filtering_20(chroma,cnt=5):
    #     recursive_cnt = cnt
    #     # tensor = np.array([[3,-1,-1],[-1,3,-1],[-1,-1,3]])
    #     chroma_copy = deepcopy(chroma)
    #     chroma = np.zeros((len(chroma),(len(chroma))))
    #
    #     for rownum in range(len(chroma) - 18):
    #         for colnum in range(len(chroma) - 18):
    #             chroma[rownum:rownum + 19, colnum:colnum + 19] += Tensor_20(chroma_copy[rownum:rownum + 19, colnum:colnum + 19])
    #     for rownum in range(len(chroma)):
    #         for colnum in range(len(chroma)):
    #             chroma[rownum, colnum] = 0 if chroma[rownum, colnum] <= 0 else chroma[rownum, colnum]
    #     if cnt == 0 :
    #         return chroma
    #     print('cnt=',recursive_cnt,'chroma',chroma)
    #     return Filtering_20(chroma, cnt= recursive_cnt-1)

#     chroma = Filtering_10(chroma)
#     for idx in range(len(chroma)):
#         chroma[idx][idx] = 0
#     print(chroma)
#     ######################################
#     plt.figure(figsize=(10, 10))
#     ld.specshow(chroma, y_axis='time', x_axis='time')
#     plt.colorbar()
#     plt.title('{}'.format(music_name))
#     plt.tight_layout()
#     plt.show()
#
#     highlight = []
# #########################################
#     chroma_normal = (chroma-np.mean(chroma))/ np.max(chroma)
#
#
#     plt.figure(figsize=(10, 10))
#     ld.specshow(chroma_normal, y_axis='time', x_axis='time')
#     plt.colorbar()
#     plt.title('{}'.format(music_name))
#     plt.tight_layout()
#     plt.show()
#     #################################################
#     print(chroma_normal)
#     print((np.mean(chroma)/np.max(chroma)))
#     print(len(chroma))
#     converttime = 0.58


    # for rownum in range(len(chroma)):
    #     for colnum in range(rownum):
    #         if chroma_normal[rownum][colnum] != 0 and rownum != colnum and chroma_normal[rownum][colnum] >= 0.9*np.max(chroma_normal):
    #             highlight.append([rownum* converttime,colnum * converttime])
    #
    # print(highlight)

# [[15.079999999999998, 4.64], [15.659999999999998, 5.22], [16.24, 5.8], [16.82, 6.38], [17.4, 6.959999999999999], [17.98, 7.539999999999999], [132.23999999999998, 15.079999999999998], [132.82, 15.659999999999998], [133.39999999999998, 16.24], [133.98, 16.82], [134.56, 17.4], [135.14, 17.98], [135.72, 18.56], [136.29999999999998, 19.139999999999997], [136.88, 19.72], [137.45999999999998, 20.299999999999997], [142.67999999999998, 4.06], [143.26, 4.64], [143.84, 5.22], [144.42, 5.8], [145.0, 6.38], [145.57999999999998, 6.959999999999999]]
# loser [[129.92, 12.76], [130.5, 13.34], [131.07999999999998, 13.919999999999998], [131.66, 14.499999999999998], [132.23999999999998, 15.079999999999998], [132.23999999999998, 68.44], [132.82, 15.659999999999998], [132.82, 69.02], [133.39999999999998, 16.24], [133.39999999999998, 69.6], [133.98, 16.82], [133.98, 70.17999999999999], [134.56, 17.4], [134.56, 70.75999999999999], [135.14, 17.98], [135.14, 71.33999999999999], [135.72, 18.56], [135.72, 71.92], [136.29999999999998, 19.139999999999997], [136.29999999999998, 72.5], [136.88, 19.72], [136.88, 73.08], [137.45999999999998, 20.299999999999997], [137.45999999999998, 73.66], [138.04, 20.88], [138.04, 74.24], [138.62, 21.459999999999997], [138.62, 74.82], [139.2, 22.04], [139.2, 75.39999999999999], [139.78, 22.619999999999997], [139.78, 75.97999999999999], [140.35999999999999, 23.2], [140.35999999999999, 76.55999999999999], [140.94, 23.779999999999998], [140.94, 77.14], [141.51999999999998, 77.72], [142.1, 78.3], [142.67999999999998, 78.88], [143.26, 79.46]]
# raindrop [[28.999999999999996, 16.82], [29.58, 17.4], [30.159999999999997, 17.98], [30.74, 18.56], [31.319999999999997, 19.139999999999997], [31.9, 19.72], [32.48, 20.299999999999997]]
#












