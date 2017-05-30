from librosa import load, stft, feature, get_duration
import librosa.display as ld
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
# superfantastic / aoa / liz / ending /
# butterfly / oohahh / seethrough / primary /

############### 플레이 정보 ###############
music_name = 'loser'           # 노래 제목
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

    def LoadSong(self):
        y, sr = load(r'c:\python\data\music\{}.mp3'.format(self.music), sr=882)
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

    def MakeNodes(self):
        cs = self.LoadSong()
        converttime = (self.time / len(cs))
        ifcondition = self.IfCondition()
        trycnt = 0

        self.FindNodes(cs, converttime, ifcondition, accuracy=0.99)

        while len(self.result) <= 1 :
            trycnt += 1
            self.FindNodes(cs, converttime, ifcondition, accuracy=1 - 0.006 * self.fibo(trycnt))
            print('Changing Accuracy...')

        print('Making Nodes Finished(2/3)')
        return cs

    def Analysis(self):
        cs = self.MakeNodes()

        if self.show == True:
            plt.figure(figsize=(10, 10))
            ld.specshow(cs, y_axis ='time', x_axis='time')
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
        return max(centrality, key=centrality.get) - 0.2


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
    # Play.PlaySong()
    song = Song(music_name)
    # for idx in range(100):

    chroma = song.LoadSong()
    from copy import deepcopy
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
    def Tensor(chroma):
        tensor = np.zeros((3,3))
        plus = (chroma[0][0] + chroma[1][1] + chroma[2][2])
        minus = (chroma[0][1] + chroma[0][2] + chroma[1][0]+chroma[1][2] + chroma[2][0] + chroma[2][1])
        tensor[1][1] = (plus -0.5*minus)/3
        return tensor

    def Tensor_10(chroma):
        tensor = np.zeros((9,9))
        plus = (chroma[0][0] + chroma[1][1] + chroma[2][2] + chroma[3][3]+ chroma[4][4]+ chroma[5][5]+ chroma[6][6]+ chroma[7][7]+ chroma[8][8])
        tensor[4][4]=((10/9) * plus - (1/9) * np.sum(chroma))/9
        return tensor

    def Filtering(chroma,cnt=20):
        recursive_cnt = cnt
        # tensor = np.array([[3,-1,-1],[-1,3,-1],[-1,-1,3]])
        chroma_copy = deepcopy(chroma)
        chroma = np.zeros((len(chroma),(len(chroma))))

        for rownum in range(len(chroma) - 2):
            for colnum in range(len(chroma) - 2):
                chroma[rownum:rownum + 3, colnum:colnum + 3] += Tensor(chroma_copy[rownum:rownum + 3, colnum:colnum + 3])
        for rownum in range(len(chroma)):
            for colnum in range(len(chroma)):
                chroma[rownum, colnum] = 0 if chroma[rownum, colnum] <= 0 else chroma[rownum, colnum]
        if cnt == 0 :
            return chroma
        print('cnt=',recursive_cnt,'chroma',chroma)
        return Filtering(chroma, cnt= recursive_cnt-1)
    def Filtering_10(chroma,cnt=10):
        recursive_cnt = cnt
        # tensor = np.array([[3,-1,-1],[-1,3,-1],[-1,-1,3]])
        chroma_copy = deepcopy(chroma)
        chroma = np.zeros((len(chroma),(len(chroma))))

        for rownum in range(len(chroma) - 8):
            for colnum in range(len(chroma) - 8):
                chroma[rownum:rownum + 9, colnum:colnum + 9] += Tensor_10(chroma_copy[rownum:rownum + 9, colnum:colnum + 9])
        for rownum in range(len(chroma)):
            for colnum in range(len(chroma)):
                chroma[rownum, colnum] = 0 if chroma[rownum, colnum] <= 0 else chroma[rownum, colnum]
        if cnt == 0 :
            return chroma
        print('cnt=',recursive_cnt,'chroma',chroma)
        return Filtering(chroma, cnt= recursive_cnt-1)


    chroma = Filtering_10(chroma)
    for idx in range(len(chroma)):
        chroma[idx][idx] = 0
    print(chroma)
    ######################################
    plt.figure(figsize=(10, 10))
    ld.specshow(chroma, y_axis='time', x_axis='time')
    plt.colorbar()
    plt.title('{}'.format(music_name))
    plt.tight_layout()
    plt.show()

    highlight = []
#########################################
    chroma_normal = (chroma-np.mean(chroma))/ np.max(chroma)


    plt.figure(figsize=(10, 10))
    ld.specshow(chroma_normal, y_axis='time', x_axis='time')
    plt.colorbar()
    plt.title('{}'.format(music_name))
    plt.tight_layout()
    plt.show()
    #################################################
    print(chroma_normal)
    print((np.mean(chroma)/np.max(chroma)))
    print(len(chroma))
    converttime = 0.58
    for rownum in range(len(chroma)):
        for colnum in range(rownum):
            if chroma_normal[rownum][colnum] != 0 and rownum != colnum and chroma_normal[rownum][colnum] >= 0.8*np.max(chroma_normal):
                highlight.append([rownum* converttime,colnum * converttime])

    print(highlight)
