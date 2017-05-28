from librosa import load, stft, feature, get_duration
import librosa.display as ld
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import networkx as nx
from pygame import mixer,init, display, time, quit
from ctypes import windll
import operator
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from collections import Counter


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
        print(chroma)
        chromaT=np.transpose(chroma,axes=(1,0))
        print('Loading Finished(1/3)')
        return cosine_similarity(chromaT)

    # def IfCondition(self):
    #     temp = []
    #     for i in range(10):
    #         temp.append('cs[m+{}][n+{}]'.format(2*i,2*i))
    #     return ' + '.join(temp)

    def FindNodes(self, cs, converttime):
        for m in range(len(cs)):
            for n in range(len(cs)):
                if (int(converttime*m) <= converttime*len(cs)-10 and int(converttime*m) >= 10):
                    if (int(converttime * n) <= converttime * len(cs) - 10 and int(converttime * n) >= 10):
                        self.result.append((int(converttime * m),int(converttime * n)))
                        self.nodes.append((int(converttime * m)))
                        # self.nodes.append((int(converttime * n)))

    def MakeNodes(self):
        cs = self.LoadSong()
        converttime = (self.time / len(cs))

        self.FindNodes(cs, converttime)

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
        # print('nodes',list(set(self.nodes)))
        G.add_nodes_from(list(set(self.nodes)))
        # print('result',self.result)
        G.add_edges_from(self.result)

        ########## eigenvector_centrality ##########
        centrality = nx.eigenvector_centrality_numpy(G)
        # centrality2 = sorted(centrality.items(), key=operator.itemgetter(1), reverse=True)
        # print(centrality2)
        print('Analyzing Finished(3/3)')

        centrality2 = list(centrality.items())
        print(centrality2)
        norm_c=normalize(centrality2)
        print('norm',norm_c)
        # return max(centrality, key=centrality.get) - 0.2
        Labels = KMeans(n_clusters=10).fit(norm_c).labels_
        min_highlight = 0
        print(Labels)
        print(Counter(Labels))
        print('dkjaljdf',np.where(Labels == 2))

        # for i in range(centrality2):
        #     i[0]
        #

        # print(max(Labels, key=Labels.get))
        # print(type(Labels))
        # LABELS = [[0.5*idx + 10, label] for idx, label in enumerate[Labels]]
        # song_dic = dict(enumerate(Labels,10))
        # print(song_dic)

        # Ks = range(1, 5)
        # km = [KMeans(n_clusters=i) for i in Ks]
        # score = [km[i].fit(centrality2).score(centrality2) for i in range(len(km))]
        # labels = [km[i].fit(centrality2).labels_ for i in range(len(km))]

        # kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
        # print(score)
        # print(labels)
        # sorted(jewelrydic.items(), key=lambda kv : (kv[1]/kv[0], -kv[0]), reverse=True)
        # cluster_array = [km[i].fit(my_matrix)]

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








