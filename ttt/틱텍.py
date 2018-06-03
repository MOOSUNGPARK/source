#######################################################################################
# Game : Mit Tic Tac Toe 완성 최종 코드
# Date : 2017.05.12
# Created by MIT university
#######################################################################################


import random
from copy import copy, deepcopy

EMPTY = 0

PLAYER_X = 1

PLAYER_O = 2

DRAW = 3

BOARD_FORMAT = "----------------------------\n| {0} | {1} | {2} |\n|--------------------------|\n| {3} | {4} | {5} |\n|--------------------------|\n| {6} | {7} | {8} |\n----------------------------"

NAMES = [' ', 'X', 'O']


def printboard(state):
    cells = []
    for i in range(3):
        for j in range(3):
            cells.append(NAMES[state[i][j]].center(6))
    print(BOARD_FORMAT.format(*cells))


def emptystate():
    return [[EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY]]


def gameover(state):
    for i in range(3):
        if state[i][0] != EMPTY and state[i][0] == state[i][1] and state[i][0] == state[i][2]:
            return state[i][0]
        if state[0][i] != EMPTY and state[0][i] == state[1][i] and state[0][i] == state[2][i]:
            return state[0][i]
    if state[0][0] != EMPTY and state[0][0] == state[1][1] and state[0][0] == state[2][2]:
        return state[0][0]
    if state[0][2] != EMPTY and state[0][2] == state[1][1] and state[0][2] == state[2][0]:
        return state[0][2]
    for i in range(3):
        for j in range(3):
            if state[i][j] == EMPTY:
                return EMPTY
    return DRAW


class Human(object):
    def __init__(self, player):
        self.player = player

    def action(self, state):
        action = None
        while action not in range(1, 10):
            action = int(input('Your move?'))
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

        printboard(state)
        if winner != EMPTY:
            return winner


class Sarsa_Agent(object):
    def __init__(self, player, lossval=0, learning_val=True):
        self.values = {}
        self.player = player
        self.lossval = lossval
        self.learning_val = learning_val
        self.epsilon = 0.1 if self.learning_val else 0
        self.alpha = 0.99
        self.gamma = 0.99
        self.prevstate = None
        self.prevscore = 0

    def statetuple(self, state):
        return (tuple(state[0]), tuple(state[1]), tuple(state[2]))

    def lookup(self, state):
        key = self.statetuple(state)
        if not key in self.values:
            self.add(key)
        return self.values[key]

    def add(self, state):
        winner = gameover(state)
        tup = self.statetuple(state)
        self.values[tup] = self.winnerval(winner)

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
        self.learning(self.winnerval(winner))
        self.prevstate = None
        self.prevscore = 0

    def action(self, state):
        r = random.random()
        if r < self.epsilon:
            move = self.random(state)
        else:
            move = self.greedy(state)

        state[move[0]][move[1]] = self.player
        self.learning(self.lookup(state))
        self.prevstate = self.statetuple(state)
        self.prevscore = self.lookup(state)
        state[move[0]][move[1]] = EMPTY
        return move

    def random(self, state):
        available = []
        for i in range(3):
            for j in range(3):
                if state[i][j] == EMPTY:
                    available.append((i, j))
        return random.choice(available)

    def greedy(self, state):
        maxval = -50000
        maxmove = None
        for i in range(3):
            for j in range(3):
                if state[i][j] == EMPTY:
                    state[i][j] = self.player
                    val = self.lookup(state)
                    state[i][j] = EMPTY
                    if val > maxval:
                        maxval = val
                        maxmove = (i, j)
        return maxmove

    def learning(self, score):
        if self.prevstate != None:
            self.values[self.prevstate] += self.alpha * ((self.gamma * score) - self.prevscore)




class QLearning_Agent(object):
    def __init__(self, player, verbose=False, lossval=0, learning_val=True):
        self.values = {}
        self.player = player
        self.lossval = lossval
        self.learning_val = learning_val
        self.epsilon = 0.1
        self.alpha = 0.99
        self.gamma = 0.99
        self.prevstate = None
        self.prevscore = 0

    def episode_over(self, winner):
        self.learning(self.winnerval(winner))
        self.prevstate = None
        self.prevscore = 0

    def action(self, state):
        r = random.random()
        if r < self.epsilon:
            move = self.random(state)
            learning_move = self.greedy(state)
        else:
            move = self.greedy(state)
            learning_move = move

        state[learning_move[0]][learning_move[1]] = self.player
        self.learning(self.lookup(state))
        state[learning_move[0]][learning_move[1]] = EMPTY

        state[move[0]][move[1]] = self.player
        self.prevstate = self.statetuple(state)
        self.prevscore = self.lookup(state)
        state[move[0]][move[1]] = EMPTY
        return move

    def random(self, state):
        available = []
        for i in range(3):
            for j in range(3):
                if state[i][j] == EMPTY:
                    available.append((i, j))
        return random.choice(available)

    def greedy(self, state):
        maxval = -50000
        maxmove = None
        for i in range(3):
            for j in range(3):
                if state[i][j] == EMPTY:
                    state[i][j] = self.player
                    val = self.lookup(state)
                    # if self.learning_val == False:
                    #     print(str(i * 3 + j + 1), str(val))
                    state[i][j] = EMPTY
                    if val > maxval:
                        maxval = val
                        maxmove = (i, j)
        return maxmove

    def learning(self, nextval):
        if self.prevstate != None:
            self.values[self.prevstate] += self.alpha * ((self.gamma * nextval) - self.prevscore)

    def lookup(self, state):
        key = self.statetuple(state)
        if not key in self.values:
            self.add(key)
        return self.values[key]

    def add(self, state):
        winner = gameover(state)
        tup = self.statetuple(state)
        self.values[tup] = self.winnerval(winner)

    def winnerval(self, winner):
        if winner == self.player:
            return 1
        elif winner == EMPTY:
            return 0.5
        elif winner == DRAW:
            return 0
        else:
            return self.lossval

    def statetuple(self, state):
        return (tuple(state[0]), tuple(state[1]), tuple(state[2]))


if __name__ == "__main__":
    p1 = Sarsa_Agent(1, lossval=-1)
    p2 = Sarsa_Agent(2, lossval=-1)
    while True:
        winner = play(p1,p2)


    p1 = Human(1)
    p2 = Human(2)
    while True:
        winner = play(p1, p2)

    p1 = Human(1)
    p2 = Human(2)
    while True:
        winner = play(p1, p2)




            #
    #
    # p1 = Sarsa_Agent(1, lossval=-1)
    # p2 = Sarsa_Agent(2, lossval=-1)
    # for i in range(10000):
    #     if i % 1000 == 0:
    #         print('Game: {0}'.format(i))
    #     winner = play(p1, p2)
    #     p1.episode_over(winner)
    #     p2.episode_over(winner)
    #     p1.epsilon = 0
    #     p2.epsilon = 0
    # while True:
    #     print(p1.epsilon)
    #     winner = play(p1, p2)
    #     p1.episode_over(winner)
    #     p2.episode_over(winner)


