from librosa import load, stft, feature, get_duration
import librosa.display as ld
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import networkx as nx
from pygame import mixer,init, display, time, quit
from ctypes import windll
from copy import deepcopy
from random import random

############### 플레이리스트 ###############
# bigbang / iluvit / inception / knock /
# palette / raindrop / rookie / russian /
# withcoffee / imissyou / ahyeah / newface /
# skyrim / noreply / soran / friday / wouldu /
# whistle / beautiful / loser / soso /
# superfantastic / liz / ending /
# butterfly / oohahh / seethrough /

############### 플레이 정보 ###############
music_name = 'bigbang'           # 노래 제목
play_duration = 10              # 재생 시간

#########################################
class Song(object):
    def __init__(self,music_name,show=False):
        self.music = music_name
        self.time = 0
        self.show = show
        self.result = []
        self.nodes = []
        self.alreadyexists = []

    def LoadSong(self):
        y, sr = load(r'c:\python\data\music\{}.mp3'.format(self.music), sr=882)
        s = np.abs(stft(y)**2)
        self.time = get_duration(y=y, sr=sr)
        chroma = feature.chroma_stft(S=s, sr=sr)
        chromaT=np.transpose(chroma,axes=(1,0))
        print('Loading Finished(1/3)')
        return cosine_similarity(chromaT)

    def FindNodes2(self,chroma, converttime):
        for rn in range(len(chroma)):
            for cn in range(rn):
                chroma[rn][cn] = 0
        chroma[chroma <= 0] = 0
        frequency = self.LineFilter(chroma)
        best_frequency = round(converttime * max(frequency, key = lambda item: item[1])[0],1)
        print('Highlight : {}m {}s'.format(int(best_frequency//60), int(best_frequency%60)))
        self.result.append(best_frequency)

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
    def LineFilter(self,chroma, line=25):
        # line = int(self.BestLine(chroma))

        frequency = []
        for cn in range(len(chroma)-line):
            # frequency.append([cn,np.count_nonzero(chroma[:][cn:cn+line])//line])
            correctcnt = 0
            for rn in range(len(chroma)-line):
                cnt = 0
                while chroma[rn+cnt][cn+cnt] != 0 and cnt < line:
                    cnt += 1

                if cnt == line:
                    correctcnt += 1

            frequency.append([cn,correctcnt])
        if max(frequency, key=lambda k:k[1])[1] >= 3 or line==0 :
            # print(max(frequency, key=lambda k:k[1])[1])
            # print(frequency)
            return frequency

        print('Auto tuning')
        shorterline = line
        return self.LineFilter(chroma, line=shorterline-5)

    def MakeNodes(self):
        chroma = self.LoadSong()
        converttime = (self.time / len(chroma))
        filtered_chroma = self.Filtering(chroma)
        filterrate = 0.25
        while filtered_chroma.all() == 0 :
            filtered_chroma = self.Filtering(chroma, filterrate= filterrate-0.05)
        self.FindNodes2(filtered_chroma, converttime)
        print('Making Nodes Finished(2/3)')
        return filtered_chroma

    def Normalization(self,chroma):
        for idx in range(len(chroma)):
            chroma[idx][idx] = 0
        return (chroma-np.mean(chroma))/np.max(chroma)

    def Tensor(self,chroma):
        tensor = np.zeros((9, 9))

        plus = (chroma[0][0] + chroma[1][1] + chroma[2][2] +
                chroma[3][3] + chroma[4][4] + chroma[5][5] +
                chroma[6][6] + chroma[7][7] + chroma[8][8])
        tensor[4][4] = ((10 / 9) * plus - (1 / 9) * np.sum(chroma)) / 9

        return tensor

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

    def Chromagram(self, chroma, show = False):
        if show == True:
            plt.figure(figsize=(10, 10))
            ld.specshow(chroma, y_axis='time', x_axis='time')
            plt.colorbar()
            plt.title('{}'.format(music_name))
            plt.tight_layout()
            plt.show()

    def Analysis(self):
        chroma = self.MakeNodes()
        self.Chromagram(chroma, show=self.show)
        return self.result[0] - 1.5

class Play(object):
    @staticmethod
    def PlaySong():
        song = Song(music_name)
        highlight = song.Analysis()

        init()
        mixer.init()
        display.set_mode((100,100))
        SetWindowPos = windll.user32.SetWindowPos
        SetWindowPos(display.get_wm_info()['window'], -1, 0, 0, 0, 0, 0x0003)
        mixer.music.load(r'c:\python\data\music\{}.mp3'.format(music_name))
        print('Music Start!')
        mixer.music.play(start=highlight)
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