# 사람 대 사람 오목 게임
import random
import csv
from copy import copy, deepcopy

EMPTY = 0  # 비어있는 칸은 0으로
DRAW = 3  # 비긴 경우는 3으로
values={}

# 9x9 오목판 만들기

BOARD_FORMAT = "------0--------1--------2--------3------" \
               "--4--------5--------6--------7--------8------\n0| {0} | {1} | {2} | {3} | {4} | {5} | {6} | {7} | {8} |\n|" \
               "----------------------------------------" \
               "--------------------------------------------\n1| {9} | {10} | {11} | {12} | {13} | {14} | {15} | {16} | {17} |\n|" \
               "----------------------------------------" \
               "--------------------------------------------\n2| {18} | {19} | {20} | {21} | {22} | {23} | {24} | {25} | {26} |\n|" \
               "----------------------------------------" \
               "--------------------------------------------\n3| {27} | {28} | {29} | {30} | {31} | {32} | {33} | {34} | {35} |\n|" \
               "----------------------------------------" \
               "--------------------------------------------\n4| {36} | {37} | {38} | {39} | {40} | {41} | {42} | {43} | {44} |\n|" \
               "----------------------------------------" \
               "--------------------------------------------\n5| {45} | {46} | {47} | {48} | {49} | {50} | {51} | {52} | {53} |\n|" \
               "----------------------------------------" \
               "--------------------------------------------\n6| {54} | {55} | {56} | {57} | {58} | {59} | {60} | {61} | {62} |\n|" \
               "----------------------------------------" \
               "--------------------------------------------\n7| {63} | {64} | {65} | {66} | {67} | {68} | {69} | {70} | {71} |\n|" \
               "----------------------------------------" \
               "--------------------------------------------\n8| {72} | {73} | {74} | {75} | {76} | {77} | {78} | {79} | {80} |\n" \
               "----------------------------------------" \
               "---------------------------------------------"
# 오목판 관련 클래스
class board():
    @staticmethod
    def emptyboard(): # 초기 바둑판 좌표값
        empty_board = [[EMPTY for i in range(9)] for i in range(9)]
        return empty_board  # [ [EMPTY,EMPTY,EMPTY... EMPTY 총 9개], [EMPTY,EMPTY,EMPTY... EMPTY 총 9개], ...]
        # [ [EMPTY x 9] x 9 ]  로 이뤄진 리스트 만들기. 즉 모든 값이 EMPTY 인 9x9의 리스트 생성

    @staticmethod
    def ttt_emptyboard(): # 초기 틱택토 바둑판 좌표값
        ttt_empty_board = [[EMPTY for i in range(4)] for i in range(4)]
        return ttt_empty_board

    @staticmethod
    def printboard(state):  # 바둑판에 돌 놓기
        Names = ['      ', '  ●  ', '  ○  '] # Names[0] -> 비어있는 경우 ' ', Names[1] -> Player1 의 돌은 흑돌, Names[2] -> Player2 의 돌은 백돌
        ball = []  # 임의의 리스트 ball 을 만들어서
        for i in range(9):
            for j in range(9):
                ball.append(Names[state[i][j]].center(3))  # 바둑판 좌표 state[i][j] 의 값이 1이면 Names[1] -> 흑돌을 리스트에 입력
        print(BOARD_FORMAT.format(*ball))  # 처음에 만든 BOARD_FORMAT 바둑판에 돌 놓기

# 머신러닝 관련 클래스
class Agent(object):
    def __init__(self, player, verbose=False, lossval=0, learning=True, epsilon=0.1, prevloc=None, loc=None, ttt=True):
        self.values = {}          # 게임결과인 가중치를 저장하는 딕셔너리
        self.player = player      # 플레이어명
        self.verbose = verbose    # 사람과 게임할 때 가중치를 바둑판에 출력해주는 변수. True 시 출력
        self.lossval = lossval    # 게임 졌을 때의 가중치로, 일반적으로 -1 값을 줌
        self.learning = learning  # 가중치를 갱신하여 학습할지 결정하는 변수. True 시 학습
        self.epsilon = epsilon    # 가중치와 상관없이 랜덤으로 수를 두는 비율. default 값은 0.1(10%)
        self.alpha = 0.99         # 게임 종료 시, 이전까지 둔 수들의 가중치를 역산할 때 사용하는 값.
                                  # (이겼을 때 마지막 수의 가중치가 1이면 그 이전 수에는 1 * 0.99, 그 이전 수에는 1 * 0.99^2 만큼 가중치 반영)
        self.prevstate = []       # 한 게임을 치르며 지금까지 뒀던 수들을 저장하는 리스트. 역산하여 가중치 줄 때 사용
        self.prevscore = 0        # 가중치값
        self.prevloc = prevloc    # 지금까지의 가중치값이 저장된 파일의 위치정보로, 누적된 가중치값을 사용하고자 할 때 입력
        self.loc = loc            # 학습 후 가중치값을 저장할 파일의 위치정보
        self.readCSV()           # readCSV() 메소드를 실행해준다! 는 의미. 사용하지 않을 때는 prevloc == None으로 설정
        self.ttt = ttt
        if self.ttt == True:
            self.tttrange = 4
        elif self.ttt == False:
            self.tttrange = 9

    def readCSV(self):            # 파일로 저장된 가중치값을 읽어들일 때 사용하는 메소드
            if self.prevloc != 'None':
                try :
                    file = open(self.prevloc, 'r')  # 파일은 __main__절에서 다음 형식으로 저장됨.
                                                    # 컬럼1 : 플레이어 정보(1or2)
                                                    # 컬럼2 : 오목판의 상태(state) 정보
                                                    # 컬럼3 : 상태 별 가중치값

                    values_list = csv.reader(file)
                    for value in values_list:
                        if int(value[0]) == self.player: # 파일 플레이어 정보가 해당 플레이어와 일치하면 다음 값들을 저장해라!라는 의미
                            try:                         # ValueError 가 나면 건너뛰고
                                self.values[tuple(eval(value[1]))] = round(float(value[2]),5)
                                                         # 가중치 딕셔너리( self.values ) 에
                                                         # 오목판 상태 정보를 key로, 가중치값을 value로 저장
                            except ValueError:
                                continue
                except FileNotFoundError:
                    values = {}

    def saveCSV(self, loc, values):
        if not loc is None:
            f = open((loc), 'a')
            w = csv.writer(f, delimiter=',', lineterminator='\n')

            for key in values:
                if self.values[key] != 0.5:
                    w.writerow([self.player, key, self.values[key]])\

            f.close()

    def Action(self, state, ttt_state=None):    # 인공지능 플레이어의 행동 기준을 설정한 메소드
        r = random.random()     # 0~1 사이의 값 랜덤 출력
        if r < self.epsilon:    # 가중치와 무관하게 랜덤하게 수를 둘 비율(self.epsilon)보다 작은 값이 출력된 경우
                                # 즉, epsilon이 0.1일 때는 10%의 확률로 랜덤하게 수를 두라는 실행절
            move = self.random(state)
            self.log('>>>>>>> Exploratory action: ' + str(move))
        elif self.ttt == True:                   # 그렇지 않을 경우 정상적으로 가장 좋은 가중치값을 고려하여 수를 두라는 실행절
            move = self.greedy(state)
            self.log('>>>>>>> Best action: ' + str(move)) # self.log 메소드는 디버깅용, 출력용 메소드로 크게 중요x
        elif self.ttt == False :
            move = self.convertgreedy(state, ttt_state)
            self.log('>>>>>>> Best action: ' + str(move))  # self.log 메소드는 디버깅용, 출력용 메소드로 크게 중요x

        if self.ttt == True :
            state[move[0]][move[1]] = self.player  # move 는 튜플 변수로, 바둑판의 정보가 담겨 있음. 예) move = (0,0) --> 바둑판 1행 1열 정보
            self.prevstate.insert(0,self.statetuple(state)) # 수를 결정한 다음에는 지금까지 뒀던 수를 저장한 prevstate 에 저장.
                                                            # 이때 최근 수를 뒤가 아닌 앞에 붙여넣음. (최근 수일수록 리스트의 왼쪽에 위치)
            self.prevscore = self.lookup(state)             # self.lookup() 메소드는 해당 오목판 상태(state)의 가중치를 출력하는 메소드로
                                                            # 지금 둔 수의 가중치를 prevscore에 저장해줌
            # print(state)
            state[move[0]][move[1]] = EMPTY                 # 다시 상태 변수(state) 초기화하는 이유 : play()함수에서 입력해주므로
                                                            # 그렇다면 왜 여기서 상태 변수에 값을 굳이 넣어줬다가 초기화했는가? prevstate, prevscore 값을 입력하기 위해
        return move                                     # self.action() 메소드는 이번 턴에 둘 수의 좌표(move)를 출력

    def random(self, state):                       # 수를 랜덤하게 둘 때 쓰는 메소드
        available = []
        tttrange = self.tttrange
        for i in range(tttrange):
            for j in range(tttrange):
                if state[i][j] == EMPTY:
                    available.append((i, j))
        return random.choice(available)

    def greedy(self, state):   # 수를 정상적으로 가중치를 고려하여 둘 때 쓰는 메소드
                                             # 크게 근접한 수를 두는지(efficiency=True) 아닌지로 나뉘어짐
        tttrange = self.tttrange
        maxval = None                              # 이번턴에 둘 수 있는 수들의 가중치 중 최대값
        maxmove = None                             # 이번 턴에서 최대 가중치를 얻게 해주는 최적의 수 좌표(move)
        maxdic = {}                                # 이번 턴에 둘 수 있는 모든 수들의 좌표(key)와 가중치(value)를 저장하는 딕셔너리
        maxlist = []                               # 최대 가중치 값을 얻게 하는 수가 여러개일 때, 이들을 저장하는 리스트
                                                   # 다수의 최적의 수 좌표에서 random 하게 maxmove를 선택하기 위해 필요

        for i in range(tttrange):
            for j in range(tttrange):
                if state[i][j] == EMPTY:       # 아직 아무도 안 뒀다면(내가 이번 턴에 둘 수 있는 수라면)
                    state[i][j] = self.player  # 해당 좌표에 수를 뒀다고 가정할때,(미리 한번 값을 입력해보고)
                    val = self.lookup(state)   # 가중치값을 self.lookup 메소드로 불러와서 val 에 저장하고
                    state[i][j] = EMPTY        # 다시 입력한 값을 지워버린다
                    maxdic[(i,j)] = val        # 그리고 maxdic 딕셔너리에 해당 좌표를 key 로 삼고 가중치를 value 로 삼아서 저장

        for key in maxdic:                           # maxdic 의 키를 모두 뽑아내서
            if maxdic[key] == max(maxdic.values()):  # 해당 키의 가중치 값이 최대 가중치 값인 경우
                maxlist.append(key)                  # 해당 키(최대 가중치를 얻게 해주는 최적의 수의 좌표)를 maxlist에 저장
        maxmove = random.choice(maxlist)             # 최적의 수가 다수라면 그중 랜덤으로 이번 턴의 maxmove 값을 선택
        maxval = max(maxdic.values())                # 그리고 이떄의 최대 가중치를 maxval 로 저장

        self.backup(maxval)                          # self.backup() 메소드는 현재의 가중치를 역산하여 지금까지 뒀던 수들의 가중치값을 갱신해주는 메소드
        return maxmove                               # self.backup() 메소드로 가중치값들을 한번 갱신해준 다음, 최적의 수의 좌표(maxmove)를 출력

    def convertgreedy(self, state, ttt_state):
        # maxdic = {}
        maxlist = []
        stateidx = []
        statelist = []
        dangerlist = []
        gameoverlist = []
        finallist = []

        for i in range(6):
            for j in range(6):
                stateidx.append((i, j))

        for idx in stateidx:
            maxdic = {}
            for i in range(4):
                for j in range(4):
                    statelist.append((idx, (i, j), self.convertmove((idx, (i, j)))))
                    ttt_state[i][j] = deepcopy(state[self.convertmove((idx,(i,j)))[0]][self.convertmove((idx,(i,j)))[1]])

            for sl in statelist:
                if ttt_state[sl[1][0]][sl[1][1]] == EMPTY:  # 아직 아무도 안 뒀다면(내가 이번 턴에 둘 수 있는 수라면)
                    ttt_state[sl[1][0]][sl[1][1]] = self.player  # 해당 좌표에 수를 뒀다고 가정할때,(미리 한번 값을 입력해보고)
                    val = self.lookup(ttt_state)  # 가중치값을 self.lookup 메소드로 불러와서 val 에 저장하고
                    ttt_state[sl[1][0]][sl[1][1]] = EMPTY  # 다시 입력한 값을 지워버린다
                    maxdic[sl[1]] = val  # 그리고 maxdic 딕셔너리에 해당 좌표를 key 로 삼고 가중치를 value 로 삼아서 저장
            # print(idx,maxdic)

            for key in maxdic:  # maxdic 의 키를 모두 뽑아내서
                statecheck = deepcopy(state)
                statecheck[self.convertmove((idx, key))[0]][self.convertmove((idx, key))[1]] = 1
                if game.gameover(statecheck) not in [0,3]:
                    gameoverlist.append(self.convertmove((idx, key)))
                statecheck[self.convertmove((idx, key))[0]][self.convertmove((idx, key))[1]] = 2
                if game.gameover(statecheck) not in [0,3]:
                    gameoverlist.append(self.convertmove((idx, key)))
                statecheck[self.convertmove((idx, key))[0]][self.convertmove((idx, key))[1]] = 0

                statecheck2 = deepcopy(ttt_state)
                gameovercnt = deepcopy(game.ttt_gameovercount(statecheck2))
                statecheck2[key[0]][key[1]] = 1
                if gameovercnt < game.ttt_gameovercount(statecheck2):
                    dangerlist.append(self.convertmove((idx, key)))
                    # print(idx,key,self.convertmove((idx, key)))
                statecheck2[key[0]][key[1]] = 2
                if gameovercnt < game.ttt_gameovercount(statecheck2):
                    dangerlist.append(self.convertmove((idx, key)))
                    # print(idx, key, self.convertmove((idx, key)))
                statecheck2[key[0]][key[1]] = 0


                if maxdic[key] == max(maxdic.values()):  # 해당 키의 가중치 값이 최대 가중치 값인 경우
                    maxlist.append(self.convertmove((idx, key)))  # 해당 키(최대 가중치를 얻게 해주는 최적의 수의 좌표)를 maxlist에 저장


        if gameoverlist != []:
            # print('gameover',gameoverlist)
            return random.choice(gameoverlist)
        if dangerlist != []:
            # print('danger',dangerlist)
            return random.choice(dangerlist)

        maxcount = dict((i, maxlist.count(i)) for i in set(maxlist))

        for key in maxcount:
            if maxcount[key] == max(maxcount.values()):
                # print('random')
                finallist.append(key)
        return random.choice(finallist)

    def backup(self, nextval):                       # 지금 둔 수의 가중치값을 역산하여 이전까지 둔 수의 가중치값을 갱신해주는 메소드
        cnt=0
        if self.ttt == True:
            for key in self.prevstate:                   # 지금까지 둔 수들의 좌표(prevstate)를 불러내서
                if self.prevstate != None and self.learning: # 지금까지 둔 수가 있고, 학습 중이라면
                    cnt += 1
                    self.values[key] += round((self.alpha ** cnt) * (nextval - self.prevscore),5) # 이렇게 가중치값들을 갱신

    def lookup(self, state):                         # 해당 좌표의 가중치값을 출력하는 메소드
        key = self.statetuple(state)
        if not key in self.values:                   # 만약 해당 좌표가 values딕셔너리에 없다면
            self.add(key)                            # 좌표를 딕셔너리에 저장해주면 되지
        return self.values[key]

    def add(self, state):                            # 좌표가 values 딕셔너리에 없는 경우 좌표와 값을 저장해주는 메소드
        # if self.ttt == True :
        #     winner = game.ttt_gameover(state)
        #     tup = self.statetuple(state)
        #     self.values[tup] = self.winnerval(winner)
        # elif self.ttt == False:
        winner = game.ttt_gameover(state)
        tup = self.statetuple(state)
        self.values[tup] = self.winnerval(winner)

    def winnerval(self, winner):                     # 게임 종료시 보상값을 설정한 메소드
        if winner == self.player:                    # 이기면 1점
            return 1
        elif winner == EMPTY:                        # 아직 경기 안 끝났을 때는 0.5점
            return 0.5
        elif winner == DRAW:                         # 비겼을 때는 0점
            return 0
        else:                                        # 지면 self.lossval점. default 값은 -1점
            return self.lossval


    def statetuple(self,state):                      # 리스트 타입인 state 변수를 values 딕셔너리의 key의 타입인 튜플 변수로 만들어주는 메소드
        tuple_list = []
        tttrange = self.tttrange
        for i in range(tttrange):
            tuple_list.append('tuple(state[{}])'.format(i))
        return (eval(', '.join(tuple_list)))

    def convertmove(self,move):                     # 4x4 좌표를 9x9 좌표로 바꿔주기
        return (move[0][0] + move[1][0], move[0][1] + move[1][1])


    def episode_over(self, winner):                  # 한 경기 끝난 경우의 메소드
        self.backup(self.winnerval(winner))          # 경기의 보상을 역산해주고
        self.prevstate = []                          # 지금까지 뒀던 수와 가중치를 초기화해줌
        self.prevscore = 0

    def log(self, s):                                # 디버깅용. 크게 중요x
        if self.verbose:
            print(s)

# 랜덤으로 두는 컴퓨터 관련 클래스
class Computer(object):
    def __init__(self, player, loc = None, values = None, ttt=False):
        self.player = player  # 컴퓨터가 두번째 두면 player 에 2 담김
        self.loc = loc
        self.values = values
        self.ttt= ttt

    def random(self, state):
        available = []
        for i in range(9):
            for j in range(9):
                if state[i][j] == EMPTY:  # 비어있는(0) 칸을 체크해서
                    available.append([i, j])  # available 리스트에 넣기
        return random.choice(available)  # 그중 아무거나 랜덤으로 고르기

    # 컴퓨터가 착수
    def Action(self, state, ttt_state=None):
        move = self.random(state)
        return move

    # 게임 종료시 출력 문구
    def episode_over(self, winner):
        if winner == DRAW:
            print('무승부입니다.')
        else:
            print('승자는 Player{} 입니다.'.format(winner))

    def saveCSV(self, loc, values):
        if not loc is None:
            for key in values:
                if self.values[key] != 0.5:
                    Fn = (loc)
                    w = csv.writer(open(Fn, 'a'), delimiter=',', lineterminator='\n')
                    w.writerow([self.player, key, self.values[key]])


# 인간 플레이어 관련 클래스
class human():
    def __init__(self, player, ttt=False):
        self.player = player
        self.ttt = ttt

    # 돌 놓기
    def Action(self, state, ttt_state=None):
        board.printboard(state)  # 바둑판 출력
        action = None
        switch_map = {}  # 바둑판 좌표 딕셔너리
        for i in range(9):
            for j in range(9):
                switch_map[10 * i + j] = (i, j)

                # 인풋 받기
        while action not in range(89) or state[switch_map[action][0]][switch_map[action][1]] != EMPTY :
            try:
                action = int(input('Player{}의 차례입니다. '.format(self.player)))
            except ValueError:
                continue

        return switch_map[action]

    # 게임 종료시 출력 문구
    def episode_over(self, winner):
        if winner == DRAW:
            print('무승부입니다.')
        else:
            print('승자는 Player{} 입니다.'.format(winner))

# 게임 진행 및 종료 관련 클래스
class game():
    @staticmethod
    def ttt_play(p1, p2, random=False):
        ttt_state = board.ttt_emptyboard()
        for i in range(16):
            if i%2 == 0:
                move = p1.Action(ttt_state)
            else :
                move = p2.Action(ttt_state)
            ttt_state[move[0]][move[1]] = i%2 +1
            winner = game.ttt_gameover(ttt_state)
            if winner != EMPTY:
                return winner


    @staticmethod
    def play(p1, p2):  # 게임 진행
        ttt_state = board.ttt_emptyboard()
        state = board.emptyboard()

        for i in range(81):
            if i % 2 == 0:
                if p1.ttt == False:
                    move = p1.Action(state,ttt_state)
                elif p1.ttt == True:
                    move = p1.Action(ttt_state)
            elif i % 2 == 1:
                if p2.ttt == False:
                    move = p2.Action(state,ttt_state)
                elif p2.ttt == True:
                    move = p2.Action(ttt_state)
            state[move[0]][move[1]] = i % 2 + 1
            # board.printboard(state)
            winner = game.gameover(state)
            if winner != EMPTY:
                board.printboard(state)
                return winner

    @staticmethod
    def ttt_gameover(state):  # 게임이 종료되는 조건 함수 생성
        for i in range(4):
            if state[0][i] != EMPTY and state[0][i] == state[1][i] and state[0][i] == state[2][i] and state[0][i] == state[3][i] :
                return state[0][i]
            if state[i][0] != EMPTY and state[i][0] == state[i][1] and state[i][0] == state[i][2] and state[i][0] == state[i][3] :
                return state[i][0]
        if state[0][0] != EMPTY and state[0][0] == state[1][1] and state[0][0] == state[2][2] and state[0][0] == state[3][3]:
            return state[0][0]
        if state[0][3] != EMPTY and state[0][3] == state[1][2] and state[0][3] == state[2][1] and state[0][3] == state[3][0]:
            return state[0][3]

        for i in range(4):
            for j in range(4):
                if state[i][j] == EMPTY:
                    return EMPTY
        return DRAW

    @staticmethod
    def ttt_gameovercount(state):
        cnt=0
        for i in range(4):
            if state[0][i] != EMPTY and state[0][i] == state[1][i] and state[0][i] == state[2][i] and state[0][i] == state[3][i] :
                cnt += 1
            if state[i][0] != EMPTY and state[i][0] == state[i][1] and state[i][0] == state[i][2] and state[i][0] == state[i][3] :
                cnt += 1
        if state[0][0] != EMPTY and state[0][0] == state[1][1] and state[0][0] == state[2][2] and state[0][0] == state[3][3]:
            cnt += 1
        if state[0][3] != EMPTY and state[0][3] == state[1][2] and state[0][3] == state[2][1] and state[0][3] == state[3][0]:
            cnt += 1
        return cnt

    @staticmethod
    def gameover(state):  # 게임이 종료되는 조건 함수 생성
        for i in range(9):
            for j in range(9):
                try:
                    # 한쪽이 이겨서 게임 종료되는 경우

                    # 가로로 다섯칸 모두 1인 경우(player1의 흑돌이 가로로 연속 다섯칸에 놓인 경우)
                    if state[i][j] * state[i][j + 1] * state[i][j + 2] * state[i][j + 3] * state[i][j + 4] == 1:
                        return 1
                        # 가로로 다섯칸 모두 2인 경우(player2의 백돌이 가로로 연속 다섯칸에 놓인 경우)
                    if state[i][j] * state[i][j + 1] * state[i][j + 2] * state[i][j + 3] * state[i][j + 4] == 32:
                        return 2
                        # 세로로 다섯칸 모두 1인 경우(player1의 흑돌이 세로로 연속 다섯칸에 놓인 경우)
                    if state[j][i] * state[j + 1][i] * state[j + 2][i] * state[j + 3][i] * state[j + 4][i] == 1:
                        return 1
                        # 세로로 다섯칸 모두 2인 경우(player2의 백돌이 가로로 연속 다섯칸에 놓인 경우)
                    if state[j][i] * state[j + 1][i] * state[j + 2][i] * state[j + 3][i] * state[j + 4][i] == 32:
                        return 2
                        # 대각선으로 다섯칸 모두 1인 경우(player1의 흑돌이 대각선으로 연속 다섯칸에 놓인 경우)
                    if state[i][j] * state[i + 1][j + 1] * state[i + 2][j + 2] * state[i + 3][j + 3] * state[i + 4][j + 4] == 1:
                        return 1
                    if state[i][j + 4] * state[i + 1][j + 3] * state[i + 2][j + 2] * state[i + 3][j + 1] * state[i + 4][j] == 1:
                        return 1
                        # 대각선으로 다섯칸 모두 2인 경우(player2의 백돌이 대각선으로 연속 다섯칸에 놓인 경우)
                    if state[i][j] * state[i + 1][j + 1] * state[i + 2][j + 2] * state[i + 3][j + 3] * state[i + 4][j + 4] == 32:
                        return 2
                    if state[i][j + 4] * state[i + 1][j + 3] * state[i + 2][j + 2] * state[i + 3][j + 1] * state[i + 4][j] == 32:
                        return 2

                except IndexError:  # range(9)로 인덱스 범위 넘어가는 경우 continue 로 예외처리하여 에러 안 뜨게 함
                    continue

                    # 한쪽이 이겨서 게임이 종료된 경우가 아니며, 빈칸이 존재하는 경우 계속 진행
        for i in range(9):
            for j in range(9):
                if state[i][j] == EMPTY:
                    return EMPTY
                    # 한쪽이 이겨서 게임이 종료된 경우가 아니며, 빈칸도 없는 경우 비김
        return DRAW

if __name__ == '__main__':

    # 파일 저장, 불러오기 input 실행절
    if input('ttt 데이터를 Save 하시겠습니까?(True or False) ') == 'True':
        input_ttt_loc = input('ttt 데이터 저장위치를 입력해주세요(예 : C:\OMOK.csv or None 입력) ')
    else :
        input_ttt_loc = None
    input_ttt_prevloc = input('참고할 파일 위치를 입력해주세요(예: C:\OMOK.csv or None 입력) ')

    p1 = Agent(1, lossval=-1, prevloc= input_ttt_prevloc, loc = input_ttt_loc, ttt=True)
    p2 = Agent(2, lossval=-1, prevloc= input_ttt_prevloc, loc = input_ttt_loc, ttt=True)
    # p2 = Computer(2)

    # 인공지능 간의 대결
    win_1 = 0
    win_2 = 0
    win_3 = 0
    for i in range(1):
        if i % 1000 == 0:
            print('Game: {0}'.format(i))

        winner = game.ttt_play(p1, p2,random=True)
        if winner == 1 :
            win_1 +=1
        elif winner == 2 :
            win_2 +=1
        elif winner == 3:
            win_3 +=1
        p1.episode_over(winner)
        p2.episode_over(winner)

    print('p1의 승률 : ', (win_1/(win_1+win_2+win_3)) * 100)
    print('p2의 승률 : ', (win_2/(win_1+win_2+win_3)) * 100)
    print('비길 확률 : ', (win_3/(win_1+win_2+win_3)) * 100)
    # 인공지능 간의 대결 후 좌표와 가중치 값을 저장해줌.
    # 이때 저장된 좌표와 가중치값 을 바탕으로 (values 딕셔너리를 이용하여) 이후 사람과 대결함
    p1.saveCSV(p1.loc, p1.values)
    p2.saveCSV(p2.loc, p2.values)



    # 인공지능 대 랜덤 컴퓨터 대결
    # p1.ttt = False
    # p2.ttt = False
    # p1 = Computer(1)
    # p2.epsilon = 0
    # win_1 = 0
    # win_2 = 0
    # win_3 = 0
    # for i in range(100):
    #     if i % 1000 == 0:
    #         print('Game: {0}'.format(i))
    #
    #     winner = game.play(p1, p2)
    #     if winner == 1 :
    #         win_1 +=1
    #     elif winner == 2 :
    #         win_2 +=1
    #     elif winner == 3:
    #         win_3 +=1
    #     p1.episode_over(winner)
    #     p2.episode_over(winner)
    #
    # print('p1의 승률 : ', (win_1/(win_1+win_2+win_3)) * 100)
    # print('p2의 승률 : ', (win_2/(win_1+win_2+win_3)) * 100)
    # print('비길 확률 : ', (win_3/(win_1+win_2+win_3)) * 100)


    # 사람과 인공지능 대결
    while True:
        p1.ttt = False
        p2.ttt = False
        p2.verbose = True
        p2.epsilon = 0
        p1 = human(1)
        winner = game.play(p1, p2)
        p1.episode_over(winner)
        p2.episode_over(winner)













