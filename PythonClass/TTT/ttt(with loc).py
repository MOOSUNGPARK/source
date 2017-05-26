
import random, csv
from copy import copy, deepcopy

# deepcopy : 메모리를 완전히 새롭게 생성
# copy : 껍데기만 카피, 내용은 동일한 곳을 가리킴

EMPTY = 0
PLAYER_X = 1
PLAYER_O = 2
DRAW = 3
BOARD_FORMAT = "----------------------------\n| {0} | {1} | {2} |\n|--------------------------|\n| {3} | {4} | {5} |\n|--------------------------|\n| {6} | {7} | {8} |\n----------------------------"
NAMES = [' ', 'X', 'O']


# 보드 출력
def printboard(state):
    cells = []
    for i in range(3):
        for j in range(3):
            cells.append(NAMES[state[i][j]].center(6))
    print(BOARD_FORMAT.format(*cells))


# 빈 판
def emptystate():
    return [[EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY]]


def gameover(state):
    # 가로/세로로 한 줄 완성한 플레이어가 있다면 그 플레이어 리턴
    for i in range(3):
        if state[i][0] != EMPTY and state[i][0] == state[i][1] and state[i][0] == state[i][2]:
            return state[i][0]
        if state[0][i] != EMPTY and state[0][i] == state[1][i] and state[0][i] == state[2][i]:
            return state[0][i]
    # 좌우 대각선
    if state[0][0] != EMPTY and state[0][0] == state[1][1] and state[0][0] == state[2][2]:
        return state[0][0]
    if state[0][2] != EMPTY and state[0][2] == state[1][1] and state[0][2] == state[2][0]:
        return state[0][2]
    # 판이 비었는지
    for i in range(3):
        for j in range(3):
            if state[i][j] == EMPTY:
                return EMPTY
    return DRAW

# 사람
class Human(object):
    def __init__(self, player):
        self.player = player


    # 착수
    def action(self, state):
        printboard(state)
        action = None
        while action not in range(1, 10):
            action = int(input('Your move? '))
        switch_map = {
            1: (0, 0),
            2: (0, 1),
            3: (0, 2),
            4: (1, 0),
            5: (1, 1),
            6: (1, 2),
            7: (2, 0),
            8: (2, 1),
            9: (2, 2)
        }
        return switch_map[action]

    def episode_over(self, winner):
        if winner == DRAW:
            print('Game over! It was a draw.')
        else:
            print('Game over! Winner: Player {0}'.format(winner))




def play(agent1, agent2):
    state = emptystate()
    for i in range(9):
        if i % 2 == 0:
            move = agent1.action(state)
        else:
            move = agent2.action(state)
        state[move[0]][move[1]] = (i % 2) + 1
        winner = gameover(state)
        if winner != EMPTY:
            return winner
    return winner


class Computer(object):
    def __init__(self, player):
        self.player = player
        self.values = {}
        self.readCSV()       # init 할때 value 에 값 채워넣을려고 함수를 실행함
        self.verbose = True  # 자리수의 확률을 확인하기 위해 사용
        self.gamenum = 0
        self.prevstate = None
        self.prevscor = 0
        self.rownum = 0
        # print(self.values)

    def readCSV(self): # 미리 학습한 csv 파일을 읽어서 values 라는 딕셔너리 변수에 넣는 함수
        file = open("c:\\data\\ttt_learn_data.csv", 'r')
        ttt_list = csv.reader(file)
        for t in ttt_list:
            try:
                self.values[((int(t[0]) ,int(t[1]) ,int(t[2])),(int(t[3]) ,int(t[4]) ,int(t[5])) ,(int(t[6]) ,int(t[7])
                             ,int(t[8])))] = float(t[10])
            except ValueError:
                continue

    def backup(self, nextval):
        if self.prevstate != None and self.learning:
            prevval = self.values[self.prevstate]
            self.values[self.prevstate] += self.alpha * (nextval - self.prevscore)
            Fn = open("E:\Test2.csv", 'a')
            w = csv.writer(Fn, delimiter=',')
            w.writerow([self.prevstate[0][0],
                        self.prevstate[0][1],
                        self.prevstate[0][2],
                        self.prevstate[1][0],
                        self.prevstate[1][1],
                        self.prevstate[1][2],
                        self.prevstate[2][0],
                        self.prevstate[2][1],
                        self.prevstate[2][2],
                        prevval,
                        self.values[self.prevstate],
                        self.player,
                        self.gamenum,
                        self.rownum])
            Fn.close()
            self.rownum += 1

    def random(self, state):
        available = []
        for i in range(3):
            for j in range(3):
                if state[i][j] == EMPTY:
                    available.append((i ,j))
        return random.choice(available)

    def greedy(self, state):
        maxval = -50000
        maxmove = None

        if self.verbose:
            cells = []
        for i in range(3):
            for j in range(3):

                if state[i][j] == EMPTY:
                    state[i][j] = self.player
                    print('state',state)

                    val = self.lookup(state) # values 에 없으면 새로 0.5 를
                                             # values 에 넣어주고 그 값을
                    state[i][j] = EMPTY      # 다시 여기로 가져온다.
                                             # 있으면 바로 values 에서 가져온다.
                    if val > maxval:

                        maxval = val
                        print('maxval'.ljust(9), maxval)
                        maxmove = (i, j)
                        print('maxmove'.ljust(9), maxmove)
                    if self.verbose:

                        cells.append('{0:.3f}'.format(val).center(6))

                elif self.verbose:

                    cells.append(NAMES[state[i][j]].center(6))

        if self.verbose:

            print (BOARD_FORMAT.format(*cells))

        return maxmove

    def lookup(self, state):

        key = self.statetuple(state) # 리스트를 튜플로 바꿔주는 역활

        if not key in self.values:

            self.add(key)  # values 에 없으며 add 함수로 추가
        print(self.values)
        print(self.values[key])
        return self.values[key]  # 있으면 그거 리턴, 없으면 만들고 리턴

    def add(self, state):

        winner = gameover(state)

        tup = self.statetuple(state)

        self.values[tup] = self.winnerval(winner) # 1,-1,0.5, 0 (비긴것)


    def statetuple(self, state):

        return (tuple(state[0]) ,tuple(state[1]) ,tuple(state[2]))



    # 컴퓨터가 착수
    def action(self, state):
        printboard(state)
        action = None
        move = self.greedy(state)
        state[move[0]][move[1]] = self.player
        return move

    def winnerval(self, winner):

        if winner == self.player:

            return 1

        elif winner == EMPTY:

            return 0.5

        elif winner == DRAW:

            return 0

        else:

            return self.lossval



    def episode_over(self, winner):
        if winner == DRAW:
            print('Game over! It was a draw.')
        else:
            print('Game over! Winner: Player {0}'.format(winner))


if __name__ == "__main__":
    p1 = Human(1)
    p2 = Computer(2)

    while True:
        winner = play(p1, p2)
        p1.episode_over(winner)
        p2.episode_over(winner)

