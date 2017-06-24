'''
확실히 게임 엔진에 loop문이 들어가면 안되는 것 같음.
그래서 해결책이 멀티스레드를 이용해서 돌려라,
콜백함수를 써라 이거인것 같은데...
어떻게 사용하는지 모름
http://stupidpythonideas.blogspot.kr/2013/10/why-your-gui-app-freezes.html
'''


from tkinter import Frame, Canvas, Label, Button, LEFT,  ALL, Tk, TOP
import random,re,time,csv

#판정관련
#0 : 헛스윙
#0 : 파울
#1 : 단타
#2 : 2루타
#3 : 3루타
#4 : 홈런

###################################################################################################
## 기록 관련 클래스
###################################################################################################
class Record:
    def __init__(self):
        self.__hit = 0  # 안타 수
        self.__bob = 0  # 볼넷 수 융
        self.__homerun = 0  # 홈런 수
        self.__atbat = 0  # 타수
        self.__avg = 0.0  # 타율

    @property
    def hit(self):
        return self.__hit

    @hit.setter
    def hit(self, hit):
        self.__hit = hit

    @property
    def bob(self):
        return self.__bob

    @bob.setter
    def bob(self,bob):
        self.__bob = bob

    @property
    def homerun(self):
        return self.__homerun

    @homerun.setter
    def homerun(self, homerun):
        self.__homerun = homerun

    @property
    def atbat(self):
        return self.__atbat

    @atbat.setter
    def atbat(self, atbat):
        self.__atbat = atbat

    @property
    def avg(self):
        return self.__avg


    @avg.setter
    def avg(self, avg):
        self.__avg = avg


    # 타자 기록 관련 메서드
    def batter_record(self, hit, bob, homerun):
        self.hit += hit
        self.bob += bob
        self.homerun += homerun
        self.atbat += 1
        self.avg = self.hit / self.atbat


###################################################################################################
## 선수 관련 클래스
###################################################################################################
class Player:
    def __init__(self, team_name, number, name):
        self.__team_name = team_name  # 팀 이름
        self.__number = number  # 타순
        self.__name = name  # 이름
        self.__record = Record()  # 기록

    @property
    def team_name(self):
        return self.__team_name

    @property
    def number(self):
        return self.__number

    @property
    def name(self):
        return self.__name

    @property
    def record(self):
        return self.__record

    @property
    def player_info(self):
        return self.__team_name + ', ' + str(self.__number) + ', ' + self.__name

    # 선수 타율 관련 메서드
    def hit_and_run(self, hit, bob, homerun):
        self.__record.batter_record(hit, bob,homerun)


###################################################################################################
## 팀 관련 클래스
###################################################################################################
class Team:
    def __init__(self, team_name, players):
        self.__team_name = team_name  # 팀 이름
        self.__player_list = self.init_player(players)  # 해당 팀 소속 선수들 정보

    @property
    def team_name(self):
        return self.__team_name

    @property
    def player_list(self):
        return self.__player_list

    # 선수단 초기화
    def init_player(self, players):
        temp = []
        for player in players:
            number, name = list(player.items())[0]
            temp.append(Player(self.__team_name, number, name))
        return temp

    def show_players(self):
        for player in self.__player_list:
            print(player.player_info)

###################################################################################################
## 저장 및 불러오기 관련 클래스  -원주/지은
#####################################################################################################

"""
d:/data/ 폴더 만들어야 합니다.

"""
'''
제껀 없어도 돼용 - 지은
'''

class Saveandload:
    DATA_SET = 0
    FILE_PATH = 'c:/data/'
    CHECK = 0
    LOAD_YN = False
    print('LOAD_YN = ', LOAD_YN)

    @staticmethod
    def make_data_set(cnt, game_info, adv, score, batter_number):
        '''

        :param player_info: 선수정보
        :param cnt: 스트라이크, 아웃, 볼 개수
        :param game_info: 등등..
        :param adv:
        :return:

        여기서 실시간 데이터를 수집하는 곳이니까,
        선수 누적하는 거 여기에서 처리할 수 있도록 제껄 빼서 넣으시거나
        제 메소드에 넣으시거나 하면 될 것 같아요.

        '''

        DATA_SET = []
        cnt = [str(data) for data in cnt] # S B O
        game_info = [str(data) for data in game_info] # 이닝, 체인지
        adv = [str(data) for data in adv] # 어드밴스
        score = [str(data) for data in score] # 점수
        batter_number = [str(data) for data in batter_number] # 배터 순서
        DATA_SET.append([game_info, adv, cnt, score, batter_number])


        Saveandload.save(DATA_SET)
        # 여기에서 저장한 이유는, 따로 세이브 버튼 활성화 되는게 아니라서, 계속 데이터를 쓰고 지우고 때문에 하고 있어요.

    @staticmethod
    def save(DATA_SET):

        with open(Saveandload.FILE_PATH + "test.csv", "wt", encoding="utf-8") as f:
            print('여기', DATA_SET)
            for row in DATA_SET:
                for idx, value in enumerate(row, 1):
                    if idx == 1:
                        print(value)
                        f.write(value[0] + '\n')
                        f.write(value[1] + '\n')
                    if idx == 2:
                        print(value)
                        f.write(value[0] + "," + value[1] + "," + value[2] + '\n')
                    if idx == 3:
                        print(value)
                        f.write(value[0] + "," + value[1] + "," + value[2] + '\n')
                    if idx == 4:
                        print(value)
                        f.write(value[0] + "," + value[1] + '\n')
                    if idx == 5:
                        print(value)
                        f.write(value[0] + "," + value[1] + '\n')
    @staticmethod
    def load():
        # Saveandload.make_data_set()
        INNING = 0
        adv = 0
        CHANGE = 0
        STRIKE_CNT = 0  # 스트라이크 개수
        BALL_CNT = 0  # 볼 개수 융
        OUT_CNT = 0  # 아웃 개수
        SCORE = 0  # [home, away]
        BATTER_NUMBER = 0
        import csv
        f = open(Saveandload.FILE_PATH + 'test.csv')  # 파일명이 바뀌어야 할 것.
        reader = csv.reader(f, delimiter=',')
        for idx, line in enumerate(reader, 1):
            if idx == 1:
                INNING = int(line[0])
            elif idx == 2:
                CHANGE = int(line[0])

            elif idx == 3:
                adv = [int(i) for i in line]
            elif idx == 4:
                STRIKE_CNT = int(line[0])
                BALL_CNT = int(line[1])
                OUT_CNT = int(line[2])
            elif idx == 5:
                SCORE = [int(i) for i in line]
            else:
                BATTER_NUMBER = [int(i) for i in line]
        print('이거이거이거',[INNING, CHANGE, adv, STRIKE_CNT, BALL_CNT, OUT_CNT, SCORE, BATTER_NUMBER])
        return [INNING, CHANGE, adv, STRIKE_CNT, BALL_CNT, OUT_CNT, SCORE, BATTER_NUMBER]

    @staticmethod
    def load_to_start_game():
        if Game.INNING == 1 and Saveandload.LOAD_YN == True:
            temp = Saveandload.load()  # list
            # INNING = 0
            Game.INNING = temp[0]
            # CHANGE = 0  # 0 : hometeam, 1 : awayteam
            Game.CHANGE = temp[1]
            # ADVANCE = [0, 0, 0]  # 진루 상황
            Game.ADVANCE = temp[2]
            Game.STRIKE_CNT = temp[3]
            Game.BALL_CNT = temp[4]
            Game.OUT_CNT = temp[5]
            # SCORE = [0, 0]  # [home, away]
            Game.SCORE = temp[6]
            # BATTER_NUMBER = [1, 1]  # [home, away] 타자 순번
            Game.BATTER_NUMBER = temp[7]

    @staticmethod
    def load_chk():
        if Saveandload.LOAD_YN == False:
            Saveandload.LOAD_YN = True
            print(Saveandload.LOAD_YN)
        else:
            pass

"""
test.txt의 데이터 형태는

0         #이닝
1         #체인지
0,0,0     #진루상황

위처럼 저장되며, 불러올 때도 저 형태의 데이터를 idx를 참고삼아 불러옴.
"""


###################################################################################################
## 게임 관련 클래스
###################################################################################################
class Game(object):
    TEAM_LIST = {
        '한화': ({1: '정근우'}, {2: '이용규'}, {3: '송광민'}, {4: '최진행'}, {5: '하주석'}, {6: '장민석'}, {7: '로사리오'}, {8: '이양기'}, {9: '최재훈'}),
        '롯데': ({1: '나경민'}, {2: '손아섭'}, {3: '최준석'}, {4: '이대호'}, {5: '강민호'}, {6: '김문호'}, {7: '정훈'}, {8: '번즈'}, {9: '신본기'}),
        '삼성': ({1: '박해민'}, {2: '강한울'}, {3: '구자욱'}, {4: '이승엽'}, {5: '이원석'}, {6: '조동찬'}, {7: '김헌곤'}, {8: '이지영'}, {9: '김정혁'}),
        'KIA': ({1: '버나디나'}, {2: '이명기'}, {3: '나지완'}, {4: '최형우'}, {5: '이범호'}, {6: '안치홍'}, {7: '서동욱'}, {8: '김민식'}, {9: '김선빈'}),
        'SK': ({1: '노수광'}, {2: '정진기'}, {3: '최정'}, {4: '김동엽'}, {5: '한동민'}, {6: '이재원'}, {7: '박정권'}, {8: '김성현'}, {9: '박승욱'}),
        'LG': ({1: '이형종'}, {2: '김용의'}, {3: '박용택'}, {4: '히메네스'}, {5: '오지환'}, {6: '양석환'}, {7: '임훈'}, {8: '정상호'}, {9: '손주인'}),
        '두산': ({1: '허경민'}, {2: '최주환'}, {3: '민병헌'}, {4: '김재환'}, {5: '에반스'}, {6: '양의지'}, {7: '김재호'}, {8: '신성현'}, {9: '정진호'}),
        '넥센': ({1: '이정후'}, {2: '김하성'}, {3: '서건창'}, {4: '윤석민'}, {5: '허정협'}, {6: '채태인'}, {7: '김민성'}, {8: '박정음'}, {9: '주효상'}),
        'KT': ({1: '심우준'}, {2: '정현'}, {3: '박경수'}, {4: '유한준'}, {5: '장성우'}, {6: '윤요섭'}, {7: '김사연'}, {8: '오태곤'}, {9: '김진곤'}),
        'NC': ({1: '김성욱'}, {2: '모창민'}, {3: '나성범'}, {4: '스크럭스'}, {5: '권희동'}, {6: '박석민'}, {7: '지석훈'}, {8: '김태군'}, {9: '이상호'})
    }


    INNING = 1  # 1 이닝부터 시작
    CHANGE = 0  # 0 : hometeam, 1 : awayteam
    STRIKE_CNT = 0  # 스트라이크 개수
    BALL_CNT = 0 #볼 개수 융
    OUT_CNT = 0  # 아웃 개수
    ADVANCE = [0, 0, 0]  # 진루 상황
    SCORE = [0, 0]  # [home, away]
    BATTER_NUMBER = [1, 1]  # [home, away] 타자 순번

    MATRIX = 5
    LOCATION = {0: [0, 0], 1: [0, 1], 2: [0, 2], 3: [0, 3], 4: [0, 4],
                5: [1, 0], 6: [1, 1], 7: [1, 2], 8: [1, 3], 9: [1, 4],
                10: [2, 0], 11: [2, 1], 12: [2, 2], 13: [2, 3], 14: [2, 4],
                15: [3, 0], 16: [3, 1], 17: [3, 2], 18: [3, 3], 19: [3, 4],
                20: [4, 0], 21: [4, 1], 22: [4, 2], 23: [4, 3], 24: [4, 4]
                } #던지는 위치의 좌표를 리스트로 저장.
    ANNOUNCE= ''

    def __init__(self, master, game_team_list, root):
        print('Home Team : ' + game_team_list[0]+' : ', Game.TEAM_LIST[game_team_list[0]])
        print('Away Team : ' + game_team_list[1]+' : ', Game.TEAM_LIST[game_team_list[1]])
        self.__hometeam = Team(game_team_list[0], Game.TEAM_LIST[game_team_list[0]])
        self.__awayteam = Team(game_team_list[1], Game.TEAM_LIST[game_team_list[1]])
        self.root = root

    @property
    def hometeam(self):
        return self.__hometeam

    @property
    def awayteam(self):
        return self.__awayteam

    # 게임 수행 메서드
    def start_game(self):
        pass
    #     if Game.INNING <= 1: #게임을 진행할 이닝을 설정. 현재는 1이닝만 진행하게끔 되어 있음.
    #         # print('====================================================================================================')
    #         Game.ANNOUNCE = '{} 이닝 {} 팀 공격 시작합니다.'.format(Game.INNING, self.hometeam.team_name if Game.CHANGE == 0 else self.awayteam.team_name)
    #         # print('====================================================================================================\n')
    #         self.attack()
    #
    #         if Game.CHANGE == 2:  # 이닝 교체
    #             Game.INNING += 1
    #             Game.CHANGE = 0
    #         self.start_game()
    #     # print('============================================================================================================')
    #     Game.ANNOUNCE = '게임 종료!!!'
    #     # print('============================================================================================================\n')
    #     self.show_record()

    # 팀별 선수 기록 출력
    def show_record(self):
        print('===================================================================================================================')
        print('==  {} | {}  =='.format(self.hometeam.team_name.center(52, ' ') if re.search('[a-zA-Z]+', self.hometeam.team_name) is not None else self.hometeam.team_name.center(50, ' '),
                                        self.awayteam.team_name.center(52, ' ') if re.search('[a-zA-Z]+', self.awayteam.team_name) is not None else self.awayteam.team_name.center(50, ' ')))
        print('==  {} | {}  =='.format(('('+str(Game.SCORE[0])+')').center(52, ' '), ('('+str(Game.SCORE[1])+')').center(52, ' ')))
        print('===================================================================================================================')
        print('== {} | {} | {} | {} | {} | {} '.format('이름'.center(8, ' '), '타율'.center(5, ' '), '타석'.center(4, ' '), '안타'.center(3, ' '), '홈런'.center(3, ' '), '볼넷'.center(3, ' ')), end='')
        print('| {} | {} | {} | {} | {} | {} =='.format('이름'.center(8, ' '), '타율'.center(5, ' '), '타석'.center(4, ' '), '안타'.center(3, ' '), '홈런'.center(3, ' '), '볼넷'.center(3, ' ')))
        print('===================================================================================================================')

        hometeam_players = self.hometeam.player_list
        awayteam_players = self.awayteam.player_list

        for i in range(9):
            hp = hometeam_players[i]
            hp_rec = hp.record
            ap = awayteam_players[i]
            ap_rec = ap.record


            save_hp=[self.hometeam.team_name, hp.name, hp_rec.avg, hp_rec.atbat, hp_rec.hit, hp_rec.homerun, hp_rec.bob ] # 지은
            save_ap=[self.awayteam.team_name, ap.name, ap_rec.avg, ap_rec.atbat, ap_rec.hit, ap_rec.homerun, ap_rec.bob ] # 지은

            self.save_record("c:\\data\\baseball_save_result2.csv", *save_hp)   # 지은
            self.save_record("c:\\data\\baseball_save_result2.csv", *save_ap)   # 지은



            print('== {} | {} | {} | {} | {} | {} |'.format(hp.name.center(6+(4-len(hp.name)), ' '), str(hp_rec.avg).center(7, ' '),
                                                      str(hp_rec.atbat).center(6, ' '), str(hp_rec.hit).center(5, ' '), str(hp_rec.homerun).center(5, ' '), str(hp_rec.bob).center(5,' ')), end='')
            print(' {} | {} | {} | {} | {} | {} =='.format(ap.name.center(6+(4-len(ap.name)), ' '), str(ap_rec.avg).center(7, ' '),
                                                        str(ap_rec.atbat).center(6, ' '), str(ap_rec.hit).center(5, ' '), str(ap_rec.homerun).center(5, ' ') , str(ap_rec.bob).center(5, ' ')))
        print('===================================================================================================================')

    # 공격 수행 메서드
    def attack(self):
        pass

        # curr_team = self.hometeam if Game.CHANGE == 0 else self.awayteam
        # player_list = curr_team.player_list
        # MATRIX = 5
        # PITCH_LOCATION = "| " + "{:^6s} | " * MATRIX #투구 영역 융
        # PITCH_LOCATION = (PITCH_LOCATION + '\n') * MATRIX #융
        # PITCH_LOCATION = "---------" * MATRIX + "\n" + PITCH_LOCATION + "---------" * MATRIX #융
        # hit_numbers = []
        #
        #
        # self.load_record("c:\\data\\baseball_save_result.csv") # 지은
        #
        #
        #
        # if Game.OUT_CNT < 3:
        #     player = self.select_player(Game.BATTER_NUMBER[Game.CHANGE], player_list)
        #     # print('====================================================================================================')
        #     Game.ANNOUNCE += '\n' + '[{}] {}번 타자[{}] 타석에 들어섭니다.\n 현재 타석 : {}번 타자[{}], 타율 : {}, 볼넷 : {}, 홈런 : {}'.format(curr_team.team_name, player.number, player.name,player.number, player.name, player.record.avg, player.record.bob, player.record.homerun)
        #     # print('====================================================================================================\n')
        #
        #
        #     random_numbers = self.throws_numbers()  # 컴퓨터가 랜덤으로 숫자 2개 생성(구질[0](0~1), 던질위치[1](0~24))
        #     # print('== [전광판] =========================================================================================')
        #     # print('==    {}      | {} : {}'.format(Game.ADVANCE[1], self.hometeam.team_name, Game.SCORE[0]))
        #     # print('==  {}   {}    | {} : {}'.format(Game.ADVANCE[2], Game.ADVANCE[0], self.awayteam.team_name, Game.SCORE[1]))
        #     # print('== [OUT : {}, BALL : {}, STRIKE : {}]'.format(Game.OUT_CNT, Game.BALL_CNT, Game.STRIKE_CNT))
        #     # print('====================================================================================================')
        #     # print(PITCH_LOCATION.format(*[str(idx) for idx in range(26)])) #투구 영역 5 * 5 출력 융
        #     # print('====================================================================================================')
        #     # print('== 현재 타석 : {}번 타자[{}], 타율 : {}, 볼넷 : {}, 홈런 : {}'.format(player.number, player.name, player.record.avg, player.record.bob, player.record.homerun))
        #
        #     while True:
        #
        #         Main.FORB = -1
        #         Main.BALLLOC = -1
        #         Main.HITORNOT = -1
        #
        #         while True:
        #             self.root.update()
        #             if Main.HITORNOT != -1:
        #                 # hit_yn = int(input('타격을 하시겠습니까?(타격 : 1 타격안함 : 0)'))
        #                 hit_yn = Main.HITORNOT
        #                 # print(hit_yn)
        #                 break
        #
        #             else:
        #                 #print('Hit 여부 선택하세요.')
        #                 #print(Main.HITORNOT)
        #                 # self.attack()
        #                 time.sleep(0.05)
        #                 continue
        #
        #         if hit_yn == 1 :#################타격 시############################ #융
        #             while True :
        #                 self.root.update()
        #                 time.sleep(0.05)
        #                 #hit_numbers = [Main.FORB, Main.BALLLOC]
        #
        #                 if Main.FORB != -1 and Main.BALLLOC != -1 :
        #                     # print('▶ 컴퓨터가 발생 시킨 숫자 : {}\n'.format(random_numbers))
        #                     hit_numbers = [Main.FORB, Main.BALLLOC]
        #                     # print(hit_numbers)
        #                     # if self.hit_number_check(hit_numbers) is False:
        #                     #     raise Exception()
        #                     hit_cnt = self.hit_judgment(random_numbers, hit_numbers)  # 안타 판별
        #                     # print(hit_cnt,'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        #                     break
        #
        #                 # else :
        #                 #     print('== ▣ 잘못된 숫자가 입력되었습니다.')
        #                 #     print(hit_numbers)
        #                 #     print('====================================================================================================')
        #                 #     print('▶ 컴퓨터가 발생 시킨 숫자 : {}\n'.format(random_numbers))
        #                 #     continue
        #
        #             if hit_cnt[0] == 0:  # strike !!!
        #                 if hit_cnt[1] == False:#파울이 아닐 때 융
        #                     Game.STRIKE_CNT += 1
        #                     Game.ANNOUNCE = '스트라이크!!!'
        #                     if Game.STRIKE_CNT == 3:
        #                         Game.ANNOUNCE = '삼진 아웃!!!'
        #                         Game.STRIKE_CNT = 0
        #                         Game.OUT_CNT += 1
        #                         player.hit_and_run(0,0,0)
        #                         break
        #
        #
        #                 if hit_cnt[1] == True:#파울일 때
        #                     if Game.STRIKE_CNT <= 1: #스트라이크 카운트가 1 이하일때는 원래대로 진행 융
        #                         Game.STRIKE_CNT += 1
        #                         Game.ANNOUNCE = '파울!!!'
        #                         if Game.STRIKE_CNT == 3:
        #                             Game.ANNOUNCE = '삼진 아웃!!!'
        #                             Game.STRIKE_CNT = 0
        #                             Game.OUT_CNT += 1
        #                             player.hit_and_run(0, 0, 0)
        #                             break
        #
        #                     # if Game.STRIKE_CNT == 2: #스트라이크 카운트가 2일때가 문제. 2일때는 파울이어도 스트라이크 카운트가 늘어나선 안됨 융
        #                     #     Game.ANNOUNCE = '파울이므로 아웃이 아닙니다. 다시 치세요!!!!'
        #
        #             else:
        #                 Game.STRIKE_CNT = 0
        #                 if hit_cnt[0] != 4:
        #                     Game.ANNOUNCE = '{}루타!!!'.format(hit_cnt[0])
        #                     player.hit_and_run(1 if hit_cnt[0] > 0 else 0, 0, 1 if hit_cnt[0] == 4 else 0)
        #                 else:
        #                     Game.ANNOUNCE = '홈런!!!'
        #                     player.hit_and_run(1 if hit_cnt[0] > 0 else 0, 0, 1 if hit_cnt[0] == 4 else 0)
        #                 self.advance_setting(hit_cnt[0])
        #                 break
        #
        #         elif hit_yn==0:######타격안하고 지켜보기 시전########################### 융
        #             #컴퓨터가 던진 공이 볼일때 융
        #             if (random_numbers[1] >= 0 and random_numbers[1] <= 4) or (random_numbers[1] % 5 == 0) or (random_numbers[1] >= 20) or ((random_numbers[1]-4) % 5 ==0):
        #                 Game.BALL_CNT += 1
        #                 Game.ANNOUNCE = '볼 !!!!!!!!!!!!!!!!!!!!!!'
        #                 if Game.BALL_CNT == 4:
        #                     Game.ANNOUNCE = '볼넷 1루출루 !!!!!!!!!!!!!!!!!!!!!! 투수가 정신을 못차리네요!'
        #                     self.advance_setting(1,True)
        #                     Game.STRIKE_CNT = 0
        #                     Game.BALL_CNT = 0
        #                     player.hit_and_run(0,1,0)
        #                     break
        #
        #             #컴퓨터가 던진 공이 스트라이크 일 때 융
        #             if (random_numbers[1]>=6 and random_numbers[1]<=8) or (random_numbers[1]>=11 and random_numbers[1]<=13) or (random_numbers[1]>=16 and random_numbers[1]<=18):
        #                 Game.STRIKE_CNT += 1
        #                 Game.ANNOUNCE = '스트라이크!!!!!!!!!!!!!'
        #                 if Game.STRIKE_CNT ==3:
        #                     Game.ANNOUNCE = '방망이도 안 휘두르고 삼진!!!!!!!!!!!!!! 제구력이 훌륭하군요!'
        #                     Game.STRIKE_CNT = 0
        #                     Game.BALL_CNT = 0
        #                     Game.OUT_CNT += 1
        #                     player.hit_and_run(0, 0, 0)
        #                     break
        #         else :
        #             continue
        #
        #
        #     if Game.BATTER_NUMBER[Game.CHANGE] == 9:
        #         Game.BATTER_NUMBER[Game.CHANGE] = 1
        #     else:
        #         Game.BATTER_NUMBER[Game.CHANGE] += 1
        #     self.attack()
        # else:
        #     Game.CHANGE += 1
        #     Game.STRIKE_CNT = 0
        #     Game.BALL_CNT = 0
        #     Game.OUT_CNT = 0
        #     Game.ADVANCE = [0, 0, 0]

    # 진루 및 득점 설정하는 메서드
    def advance_setting(self, hit_cnt, bob=False):
        if hit_cnt == 4:  # 홈런인 경우
            Game.SCORE[Game.CHANGE] += (Game.ADVANCE.count(1)+1)
            Game.ADVANCE = [0, 0, 0]
        else:
            if bob==False: #볼넷이 아닐때
                for i in range(len(Game.ADVANCE), 0, -1):
                    if Game.ADVANCE[i-1] == 1:
                        if (i + hit_cnt) > 3:  # 기존에 출루한 선수들 중 득점 가능한 선수들에 대한 진루 설정
                            Game.SCORE[Game.CHANGE] += 1
                            Game.ADVANCE[i-1] = 0
                        else:  # 기존 출루한 선수들 중 득점권에 있지 않은 선수들에 대한 진루 설정
                            Game.ADVANCE[i-1 + hit_cnt] = 1
                            Game.ADVANCE[i-1] = 0
                Game.ADVANCE[hit_cnt-1] = 1  # 타석에 있던 선수에 대한 진루 설정


            elif bob==True: #볼넷일때
                if Game.ADVANCE[0]==1: #1루에 주자가 있을때.
                    if Game.ADVANCE[1]==0 and Game.ADVANCE[2]==1:#1,3루 일때
                        Game.ADVANCE[1]=1
                    else: #그 외의 경우
                        for i in range(len(Game.ADVANCE), 0, -1):
                            if Game.ADVANCE[i-1] == 1:
                                if (i + hit_cnt) > 3:  # 기존에 출루한 선수들 중 득점 가능한 선수들에 대한 진루 설정
                                    Game.SCORE[Game.CHANGE] += 1
                                    Game.ADVANCE[i-1] = 0
                                else:  # 기존 출루한 선수들 중 득점권에 있지 않은 선수들에 대한 진루 설정
                                    Game.ADVANCE[i-1 + hit_cnt] = 1
                                    Game.ADVANCE[i-1] = 0
                        Game.ADVANCE[hit_cnt-1] = 1  # 타석에 있던 선수에 대한 진루 설정


                else: #1루에 주자가 없을때는 1루에만 주자를 채워 넣는다.
                    Game.ADVANCE[0] = 1

    # 컴퓨터가 생성한 랜덤 수와 플레이어가 입력한 숫자가 얼마나 맞는지 판단
    def hit_judgment(self, random_ball, hit_numbers): #(공던질위치, 구질) #융
        cnt = 0
        Foul = False
        UPDOWN = abs(Game.LOCATION[random_ball[1]][0] - Game.LOCATION[hit_numbers[1]][0]) #투수와 타자의 선택한 공 위치의 높낮이차이 #융
        #UPDOWN = abs(Game.LOCATION[random_ball[1]][0] - Main.Y1)  # 투수와 타자의 선택한 공 위치의 높낮이차이 #융
        L_OR_R = abs(Game.LOCATION[random_ball[1]][1] - Game.LOCATION[hit_numbers[1]][1]) #투수와 타자의 선택한 공 위치의 좌우차이 #융
        #L_OR_R = abs(Game.LOCATION[random_ball[1]][1] - Main.X1) #투수와 타자의 선택한 공 위치의 좌우차이 #융

        if random_ball[0] == hit_numbers[0]: #투수가 던진 공의 구질과 타자가 선택한 구질이 같을 때 #융
            if random_ball[1] == hit_numbers[1]:#위치가 같으니까 홈런 #융
                cnt += 4
            elif UPDOWN == 0:#높낮이가 같은 선상일 때 #융
                if L_OR_R == 1: #좌우로 1칸 차이 #융
                    Game.ANNOUNCE = '3루타~'
                    cnt += 3
                elif L_OR_R == 2: #좌우로 2칸 차이 #융
                    Game.ANNOUNCE = '2루타~'
                    cnt += 2
                elif L_OR_R >= 3: #좌우로 3칸 차이 #융
                    Game.ANNOUNCE = '1루타~'
                    cnt += 1
            elif UPDOWN == 1:#높낮이 차이가 하나일때 #융
                if L_OR_R ==1:
                    Game.ANNOUNCE = '2루타~'
                    cnt += 2
                elif L_OR_R ==2:
                    Game.ANNOUNCE = '1루타~'
                    cnt += 1
                elif L_OR_R >= 3:
                    Game.ANNOUNCE = '파울'
                    cnt += 0
                    Foul = True
            elif UPDOWN >= 2:#높낮이가 두개이상 차이날때 #융
                Game.ANNOUNCE = '헛스윙~!'
                cnt += 0

        else: #투수가 던진 공의 구질과 타자가 선택한 구질이 다를 때 융
            if random_ball[0] == hit_numbers[0]:#위치가 같지만 구질은 다르니 3루타 융
                cnt += 3
            elif UPDOWN == 0:#높낮이가 같은 선상일 때 #융
                if L_OR_R == 1:
                    Game.ANNOUNCE = '2루타~'
                    cnt += 2
                elif L_OR_R == 2:
                    Game.ANNOUNCE = '1루타~'
                    cnt += 1
                elif L_OR_R >= 3:
                    Game.ANNOUNCE = '파울 ㅜㅜ'
                    cnt += 0
                    Foul = True
            elif UPDOWN == 1:#높낮이 차이가 하나일때 융
                if L_OR_R ==1:
                    Game.ANNOUNCE = '1루타~'
                    cnt += 1
                elif L_OR_R ==2:
                    Game.ANNOUNCE = '파울ㅠㅠ'
                    cnt += 0
                    Foul = True
                elif L_OR_R >= 3:
                    Game.ANNOUNCE = '헛스윙'
                    cnt += 0
            elif UPDOWN >= 2:#높낮이가 두개이상 차이날때 융
                Game.ANNOUNCE = '헛스윙~!'
                cnt += 0

        return cnt,Foul

    #선수가 입력한 숫자 확인
    #융
    def hit_number_check(self,hit_numbers): #구질(0~1),위치(0~24)가 들어옴 융
        if len(hit_numbers) == 2:
            if (hit_numbers[0] >= 0 and hit_numbers[0] <= 1) and (hit_numbers[1] >= 0 and hit_numbers[1] <= 24):
                return True
            else:
                return False

    # 선수 선택
    def select_player(self, number, player_list):
        for player in player_list:
            if number == player.number:
                return player

    # 랜덤으로 숫자 생성(1~20)
    def throws_numbers(self):
        while True:
            random_loc = random.randint(0, 24)  # 0 ~ 24 중에 랜덤 수를 출력
            random_ball= random.randint(0,  1)   #
            return random_ball, random_loc

class Main(Game):
    HITORNOT = -1
    FORB = -1
    BALLLOC = -1
    COLOR = ["white", "red"]

    def __init__(self, master, game_team_list, root):
        super().__init__(master,game_team_list,root)
        self.root = root
        self.game = Game(master, game_team_list, root)
        self.frame = Frame(master)
        self.frame.pack(fill="both", expand=True)
        self.canvas = Canvas(self.frame, width=1000, height=600)
        self.canvas.pack(fill="both", expand=True)
        # self.label = Label(self.frame, text='야구 게임', height=6, bg='white', fg='black')
        # self.label.pack(fill="both", expand=True)
        # self.label.place(x=0, y=0, width=1000, height=100, bordermode='outside')
        self.frameb = Frame(self.frame)
        self.frameb.pack(fill="both", expand=True)
        self.newgame = Button(self.frameb, text='New Game', height=4, command=self.start_game, bg='purple',
                              fg='white')
        self.newgame.pack(fill="both", expand=True, side=LEFT)
        self.loadgame = Button(self.frameb, text='Load Game', height=4, command=self.Loadgame, bg='white',
                               fg='purple')
        self.loadgame.pack(fill="both", expand=True, side=LEFT)
        self.hit = Button(self.frameb, text='타격', width=5, height=2, command=self.Hitbutton, bg='purple',
                          fg='white')
        self.hit.pack(fill="both", expand=True)
        self.nohit = Button(self.frameb, text='타격안함', width=5, height=2, command=self.Nohitbutton, bg='purple',
                            fg='white')
        self.nohit.pack(fill="both", expand=True, side=TOP)
        self.fastball = Button(self.frameb, text='직구', width=5, height=2, command=self.FastBall, bg='purple',
                               fg='white')
        self.fastball.pack(fill="both", expand=True, side=TOP)
        self.breakingball = Button(self.frameb, text='변화구', width=5, height=2, command=self.BreakingBall,
                                   bg='purple', fg='white')
        self.breakingball.pack(fill="both", expand=True, side=TOP)
        self.canvas.bind("<ButtonPress-1>", self.Throwandhit)
        #self.canvas.bind("<Motion>", self.board)
        self.ball_color=[]
        self.strike_color=[]
        self.out_color=[]
        self.board()

    def attack(self):
        curr_team = self.hometeam if Game.CHANGE == 0 else self.awayteam
        player_list = curr_team.player_list
        MATRIX = 5
        PITCH_LOCATION = "| " + "{:^6s} | " * MATRIX #투구 영역 융
        PITCH_LOCATION = (PITCH_LOCATION + '\n') * MATRIX #융
        PITCH_LOCATION = "---------" * MATRIX + "\n" + PITCH_LOCATION + "---------" * MATRIX #융
        hit_numbers = []


        if Game.OUT_CNT < 3:
            player = self.select_player(Game.BATTER_NUMBER[Game.CHANGE], player_list)
            # print('====================================================================================================')
            Game.ANNOUNCE += '\n' + '[{}] {}번 타자[{}] 타석에 들어섭니다.\n 현재 타석 : {}번 타자[{}], 타율 : {}, 볼넷 : {}, 홈런 : {}'.format(curr_team.team_name, player.number, player.name,player.number, player.name, player.record.avg, player.record.bob, player.record.homerun)
            # print('====================================================================================================\n')
            self.board()

            random_numbers = self.throws_numbers()  # 컴퓨터가 랜덤으로 숫자 2개 생성(구질[0](0~1), 던질위치[1](0~24))
            # print('== [전광판] =========================================================================================')
            # print('==    {}      | {} : {}'.format(Game.ADVANCE[1], self.hometeam.team_name, Game.SCORE[0]))
            # print('==  {}   {}    | {} : {}'.format(Game.ADVANCE[2], Game.ADVANCE[0], self.awayteam.team_name, Game.SCORE[1]))
            # print('== [OUT : {}, BALL : {}, STRIKE : {}]'.format(Game.OUT_CNT, Game.BALL_CNT, Game.STRIKE_CNT))
            # print('====================================================================================================')
            # print(PITCH_LOCATION.format(*[str(idx) for idx in range(26)])) #투구 영역 5 * 5 출력 융
            # print('====================================================================================================')
            # print('== 현재 타석 : {}번 타자[{}], 타율 : {}, 볼넷 : {}, 홈런 : {}'.format(player.number, player.name, player.record.avg, player.record.bob, player.record.homerun))

            while True:

                PLAYER_INFO = [player.number, player.name, player.record.avg, player.record.bob,
                                                player.record.homerun]
                CNT = [Game.STRIKE_CNT, Game.BALL_CNT, Game.OUT_CNT]
                GAME_INFO = [Game.INNING, Game.CHANGE]
                ADV = Game.ADVANCE
                SCORE = Game.SCORE
                BATTER_NUMBER = Game.BATTER_NUMBER
                print()
                Saveandload.make_data_set(CNT, GAME_INFO, ADV, SCORE, BATTER_NUMBER)

                Main.FORB = -1
                Main.BALLLOC = -1
                Main.HITORNOT = -1

                while True:
                    self.root.update()
                    if Main.HITORNOT != -1:
                        # hit_yn = int(input('타격을 하시겠습니까?(타격 : 1 타격안함 : 0)'))
                        hit_yn = Main.HITORNOT
                        # print(hit_yn)
                        break

                    else:
                        #print('Hit 여부 선택하세요.')
                        #print(Main.HITORNOT)
                        # self.attack()
                        time.sleep(0.05)
                        continue

                if hit_yn == 1 :#################타격 시############################ #융
                    while True :
                        self.root.update()
                        time.sleep(0.05)
                        #hit_numbers = [Main.FORB, Main.BALLLOC]

                        if Main.FORB != -1 and Main.BALLLOC != -1 :
                            # print('▶ 컴퓨터가 발생 시킨 숫자 : {}\n'.format(random_numbers))
                            hit_numbers = [Main.FORB, Main.BALLLOC]
                            # print(hit_numbers)
                            # if self.hit_number_check(hit_numbers) is False:
                            #     raise Exception()
                            hit_cnt = self.hit_judgment(random_numbers, hit_numbers)  # 안타 판별

                            # print(hit_cnt,'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                            break

                        # else :
                        #     print('== ▣ 잘못된 숫자가 입력되었습니다.')
                        #     print(hit_numbers)
                        #     print('====================================================================================================')
                        #     print('▶ 컴퓨터가 발생 시킨 숫자 : {}\n'.format(random_numbers))
                        #     continue

                    if hit_cnt[0] == 0:  # strike !!!
                        if hit_cnt[1] == False:#파울이 아닐 때 융
                            Game.STRIKE_CNT += 1
                            Game.ANNOUNCE = '스트라이크!!!'
                            self.board()
                            if Game.STRIKE_CNT == 3:
                                Game.ANNOUNCE = '삼진 아웃!!!'
                                Game.STRIKE_CNT = 0
                                Game.OUT_CNT += 1
                                player.hit_and_run(0,0,0)
                                break


                        if hit_cnt[1] == True:#파울일 때
                            if Game.STRIKE_CNT <= 1: #스트라이크 카운트가 1 이하일때는 원래대로 진행 융
                                Game.STRIKE_CNT += 1
                                Game.ANNOUNCE = '파울!!!'
                                self.board()
                                if Game.STRIKE_CNT == 3:
                                    Game.ANNOUNCE = '삼진 아웃!!!'
                                    self.board()
                                    Game.STRIKE_CNT = 0
                                    Game.OUT_CNT += 1
                                    player.hit_and_run(0, 0, 0)
                                    break

                            # if Game.STRIKE_CNT == 2: #스트라이크 카운트가 2일때가 문제. 2일때는 파울이어도 스트라이크 카운트가 늘어나선 안됨 융
                            #     Game.ANNOUNCE = '파울이므로 아웃이 아닙니다. 다시 치세요!!!!'

                    else:
                        Game.STRIKE_CNT = 0
                        if hit_cnt[0] != 4:
                            Game.ANNOUNCE = '{}루타!!!'.format(hit_cnt[0])
                            player.hit_and_run(1 if hit_cnt[0] > 0 else 0, 0, 1 if hit_cnt[0] == 4 else 0)
                            self.board()
                        else:
                            Game.ANNOUNCE = '홈런!!!'
                            player.hit_and_run(1 if hit_cnt[0] > 0 else 0, 0, 1 if hit_cnt[0] == 4 else 0)
                            self.board()
                        self.advance_setting(hit_cnt[0])
                        break

                elif hit_yn==0:######타격안하고 지켜보기 시전########################### 융
                    #컴퓨터가 던진 공이 볼일때 융
                    if (random_numbers[1] >= 0 and random_numbers[1] <= 4) or (random_numbers[1] % 5 == 0) or (random_numbers[1] >= 20) or ((random_numbers[1]-4) % 5 ==0):
                        Game.BALL_CNT += 1
                        Game.ANNOUNCE = '볼 !!!!!!!!!!!!!!!!!!!!!!'
                        self.board()
                        if Game.BALL_CNT == 4:
                            Game.ANNOUNCE = '볼넷 1루출루 !!!!!!!!!!!!!!!!!!!!!! 투수가 정신을 못차리네요!'
                            self.advance_setting(1,True)
                            self.board()
                            Game.STRIKE_CNT = 0
                            Game.BALL_CNT = 0
                            player.hit_and_run(0,1,0)
                            break

                    #컴퓨터가 던진 공이 스트라이크 일 때 융
                    if (random_numbers[1]>=6 and random_numbers[1]<=8) or (random_numbers[1]>=11 and random_numbers[1]<=13) or (random_numbers[1]>=16 and random_numbers[1]<=18):
                        Game.STRIKE_CNT += 1
                        Game.ANNOUNCE = '스트라이크!!!!!!!!!!!!!'
                        self.board()
                        if Game.STRIKE_CNT ==3:
                            Game.ANNOUNCE = '방망이도 안 휘두르고 삼진!!!!!!!!!!!!!! 제구력이 훌륭하군요!'
                            Game.STRIKE_CNT = 0
                            Game.BALL_CNT = 0
                            Game.OUT_CNT += 1
                            player.hit_and_run(0, 0, 0)
                            self.board()
                            break
                else :
                    continue


            if Game.BATTER_NUMBER[Game.CHANGE] == 9:
                Game.BATTER_NUMBER[Game.CHANGE] = 1
            else:
                Game.BATTER_NUMBER[Game.CHANGE] += 1
            self.attack()
        else:
            Game.CHANGE += 1
            Game.STRIKE_CNT = 0
            Game.BALL_CNT = 0
            Game.OUT_CNT = 0
            Game.ADVANCE = [0, 0, 0]
            self.board()

    def start_game(self):
        Saveandload.load_to_start_game()

        if Game.INNING <= 1: #게임을 진행할 이닝을 설정. 현재는 1이닝만 진행하게끔 되어 있음.
            # print('====================================================================================================')
            Game.ANNOUNCE = '{} 이닝 {} 팀 공격 시작합니다.'.format(Game.INNING, self.game.hometeam.team_name if Game.CHANGE == 0 else self.game.awayteam.team_name)
            # print('====================================================================================================\n')
            self.board()
            self.attack()

            if Game.CHANGE == 2:  # 이닝 교체
                Game.INNING += 1
                Game.CHANGE = 0
            self.start_game()
        # print('============================================================================================================')
        Game.ANNOUNCE = '게임 종료!!!'
        # print('============================================================================================================\n')
        self.game.show_record()

    def Loadgame(self):
        Saveandload.load_chk()
        self.start_game()

    def board(self):
        hometeam = self.game.hometeam.team_name
        awayteam = self.game.awayteam.team_name

        homescore = self.game.SCORE[0]
        awayscore = self.game.SCORE[1]
        announce = self.game.ANNOUNCE
        inning = self.game.INNING
        change = self.game.CHANGE
        attackordefence = [["공격", "수비"] if change == 0 else ["수비", "공격"]]
        scoreformat = '{} : {}  ({}) | {}이닝 | ({})  {} : {}'

        if self.game.BALL_CNT==0:
            self.ball_color=["white","white","white"]
        elif self.game.BALL_CNT==1:
            self.ball_color=["orange","white","white"]
        elif self.game.BALL_CNT==2:
            self.ball_color=["orange","orange","white"]
        elif self.game.BALL_CNT==3:
            self.ball_color=["orange","orange","orange"]

        if self.game.STRIKE_CNT==0:
            self.strike_color=['white','white']
        elif self.game.STRIKE_CNT==1:
            self.strike_color=["blue","white"]
        elif self.game.STRIKE_CNT==2:
            self.strike_color=["blue","blue"]

        if self.game.OUT_CNT==0:
            self.out_color=['white','white']
        elif self.game.OUT_CNT==1:
            self.out_color=["red","white"]
        elif self.game.OUT_CNT==2:
            self.out_color=["red","red"]

        self.canvas.create_rectangle(500, 0, 1000, 600, outline="black")
        self.canvas.create_rectangle(500, 0, 1000, 100, outline="black")
        self.canvas.create_rectangle(600, 600, 700, 0, outline="black")
        self.canvas.create_rectangle(500, 100, 1000, 200, outline="black")
        self.canvas.create_rectangle(700, 600, 800, 0, outline="black")
        self.canvas.create_rectangle(500, 200, 1000, 300, outline="black")
        self.canvas.create_rectangle(800, 600, 900, 0, outline="black")
        self.canvas.create_rectangle(500, 300, 1000, 400, outline="black")
        self.canvas.create_rectangle(900, 600, 1000, 0, outline="black")
        self.canvas.create_rectangle(500, 400, 1000, 500, outline="black")
        self.canvas.create_rectangle(500, 600, 1000, 600, outline="black")
        self.canvas.create_rectangle(0, 100, 480, 600, fill="green")

        self.canvas.create_line(240, 135, 35, 330, width=4, fill="white")
        self.canvas.create_line(240, 135, 445, 330, width=4, fill="white")
        self.canvas.create_line(40, 330, 240, 515, width=4, fill="white")
        self.canvas.create_line(445, 330, 240, 515, width=4, fill="white")

        self.canvas.create_oval(225, 120, 255, 150, fill=Main.COLOR[self.game.ADVANCE[1]])  # 2루
        self.canvas.create_oval(20, 315, 50, 345, fill=Main.COLOR[self.game.ADVANCE[2]])  # 3루
        self.canvas.create_oval(430, 315, 460, 345, fill=Main.COLOR[self.game.ADVANCE[0]])  # 1루
        self.canvas.create_oval(225, 500, 255, 530, fill="white")

        self.canvas.create_text(350, 490, font=("Courier", 12), text="B")
        self.canvas.create_oval(370, 480, 390, 500, fill=self.ball_color[0])#볼
        self.canvas.create_oval(405, 480, 425, 500, fill=self.ball_color[1])#볼
        self.canvas.create_oval(440, 480, 460, 500, fill=self.ball_color[2])#볼
        self.canvas.create_text(350, 525, font=("Courier", 12), text="S")
        self.canvas.create_oval(370, 515, 390, 535, fill=self.strike_color[0])#스트라이크
        self.canvas.create_oval(405, 515, 425, 535, fill=self.strike_color[1])  # 스트라이크
        self.canvas.create_text(350, 560, font=("Courier", 12), text="O")
        self.canvas.create_oval(370, 550, 390, 570, fill=self.out_color[0])  # 아웃
        self.canvas.create_oval(405, 550, 425, 570, fill=self.out_color[1])  # 아웃

        self.label = Label(self.frame, text=scoreformat.format(hometeam, homescore, attackordefence[0][0], inning, attackordefence[0][1], awayscore, awayteam), height=6, bg='white', fg='black')
        self.label.config(font=("Courier", 20))
        self.label.pack(fill="both", expand=True)
        self.label.place(x=0, y=0, width=1000, height=38, bordermode='outside')
        self.label = Label(self.frame, text=announce, height=6, bg='white', fg='black')
        self.label.config(font=("Courier", 10))
        self.label.pack(fill="both", expand=True)
        self.label.place(x=0, y=30, width=1000, height=70, bordermode='outside')

    def Throwandhit(self,event):
        loclist = [[5 * i + j for j in range(5)] for i in range(5)]
        for k in range(500, 1000, 100):
            for j in range(100, 600, 100):
                if event.x in range(k, k + 100) and event.y in range(j, j + 100):
                    X1 = int((k - 500) / 100)
                    Y1 = int((j - 100) / 100)
        # print('마우스 위치 좌표', X1, Y1)
        # print('리턴 좌표', loclist[Y1][X1])
        Main.BALLLOC = loclist[Y1][X1]
        self.board()

    def Hitbutton(self):
        # print('hit')
        Main.HITORNOT = 1
        self.board()

    def Nohitbutton(self):
        print('no hit')
        Main.HITORNOT = 0
        self.board()

    def FastBall(self):
        # print('Fastball')
        Main.FORB = 1
        self.board()

    def BreakingBall(self):
        # print('Brakingball')
        Main.FORB = 0
        self.board()





if __name__ == '__main__':
    # game_team_list = []
    # if game_team_list == [] :
    #     print('====================================================================================================')
    #     print('한화 / ', '롯데 / ', '삼성 / ', 'KIA / ', 'SK / ', 'LG / ', '두산 / ', '넥센 / ', 'KT / ', 'NC / ')
    #     game_team_list = input('=> 게임을 진행할 두 팀을 입력하세요 : ').split(' ')
    #     print('====================================================================================================\n')
    #     # if (game_team_list[0] in Game.TEAM_LIST) and (game_team_list[1] in Game.TEAM_LIST):
    #     #     root = Tk()
    #     #     Main(root, game_team_list)
    #     #     root.mainloop()
    #
    #     root = Tk()
    #     app = Main(root,game_team_list, root)
    #     root.mainloop()

    while True:
        game_team_list = []
        print('====================================================================================================')
        print('한화 / ', '롯데 / ', '삼성 / ', 'KIA / ', 'SK / ', 'LG / ', '두산 / ', '넥센 / ', 'KT / ', 'NC / ')
        game_team_list = input('=> 게임을 진행할 두 팀을 입력하세요 : ').split(' ')
        print('====================================================================================================\n')
        if (game_team_list[0] in Game.TEAM_LIST) and (game_team_list[1] in Game.TEAM_LIST):
            break
        else:
            print('입력한 팀 정보가 존재하지 않습니다. 다시 입력해주세요.')

    root = Tk()
    app = Main(root, game_team_list, root)
    root.mainloop()



