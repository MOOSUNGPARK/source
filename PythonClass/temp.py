from librosa import load, stft, feature, get_duration
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import scale, MinMaxScaler, normalize
import matplotlib.pyplot as plt
import networkx as nx
from pygame import mixer,init, display, time, quit
from ctypes import windll
from sklearn.cluster import KMeans



############### 플레이리스트 ###############
# bigbang / iluvit / inception / knock /
# palette / raindrop / rookie / russian /
# withcoffee / imissyou / ahyeah / newface /
# skyrim / noreply / soran / friday / wouldu /
# whistle / beautiful / loser / soso /
# superfantastic / aoa / liz / ending /
# butterfly / oohahh / seethrough / primary /

############### 플레이 정보 ###############
music_name = 'knock'           # 노래 제목
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
        s = np.abs(stft(y)**4)
        self.time = get_duration(y=y, sr=sr)
        chroma = feature.chroma_stft(S=s, sr=sr)
        chromaT= np.transpose(chroma)
        song_dic=dict(enumerate(chromaT))
        return chromaT

    def MakeNodes(self):
        chroma = self.LoadSong()

        # converttime = (self.time / len(cs))
        # kmeans = KMeans(n_clusters=3, random_state=0).fit(chroma)
        Ks = range(1, 414)
        km = [KMeans(n_clusters=i) for i in Ks]
        score = [km[i].fit(chroma).score(chroma) for i in range(len(km))]

        print(score)


        print('chroma', chroma)

        # min_max_scaler = MinMaxScaler()

        # cs = min_max_scaler.fit_transform(cs)

        print('Making Nodes Finished(2/3)')


    # def Analysis2(self):
    #     cs = self.LoadSong()
    #     print(cs)
    #     print(len(cs))
    #     print(len(cs[0]))
        #converttime = (self.time / len(cs))
        # record = []
        #
        # for idx in range(len(cs)):
        #     try :
        #         record.append([int(converttime * idx), sum(sum([cs[:,idx],cs[:,idx+1],cs[:,idx+2],cs[:,idx+3],
        #                                                     cs[:,idx+4],cs[:,idx+5],cs[:,idx+6],cs[:,idx+7],
        #                                                     cs[:,idx+8],cs[:,idx+9]]))])
        #     except IndexError:
        #         record.append([int(converttime * idx),1])
        # return cs
        # print(cs)
        # print(len(cs))
        # print(len(cs[0]))
        # print(cs[:,0])

    def Analysis(self):
        cs,volume = self.MakeNodes()
        print(cs)

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
        print(centrality)
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
    song.MakeNodes()
    # print(song.Analysis2())
    # song.Analysis2()
    # print(max(song.Analysis2(),key= lambda s:s[1]))







