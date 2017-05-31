from librosa import load, stft, feature, get_duration
import librosa.display as ld
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from pygame import mixer,init, display, time, quit
from ctypes import windll
from copy import deepcopy

############### 플레이리스트 ###############
# bigbang / iluvit / inception / knock /
# palette / raindrop / rookie / russian /
# withcoffee / imissyou / ahyeah / newface /
# skyrim / noreply / soran / friday / wouldu /
# whistle / beautiful / loser / soso /
# superfantastic / liz / ending /
# butterfly / oohahh / seethrough /
# canyoufeel / do_you_hear_the_people_sing /
# let_it_be / summer / summer_nights / time /
# tell_me_if_you / the_time_goes_on / loststars
# maroon5 / sugar / kings

############### 플레이 정보 ###############
music_name = 'the_time_goes_on'           # 노래 제목
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
        y, sr = load(r'd:\python\data\music\{}.mp3'.format(self.music), sr=882)
        s = np.abs(stft(y)**2)
        self.time = get_duration(y=y, sr=sr)
        chroma = feature.chroma_stft(S=s, sr=sr)
        chromaT=np.transpose(chroma,axes=(1,0))
        print('Loading Finished !')
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

    def LineFilter(self,chroma, line=25):
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
        if max(frequency, key=lambda k:k[1])[1] >= 3 or line==0 :
            return frequency
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

        if cnt == 0:
            return self.Normalization(chroma)
        print('Count Down', recursive_cnt)
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
        return self.result[0] - 0.5

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
        mixer.music.load(r'd:\python\data\music\{}.mp3'.format(music_name))
        print('Music Start!')
        mixer.music.play(start=highlight)
        time.wait(play_duration * 1000)
        quit()

if __name__ == '__main__':
    Play.PlaySong()
