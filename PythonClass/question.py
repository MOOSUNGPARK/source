from math import *
import random
import copy
import time

BOARD_FORMAT = "|-|---|1|------|2|------|3|------|4|------|5|------|6|------|7|------|8|------|9|---\n" \
               "|1| {0} | {1} | {2} | {3} | {4} | {5} | {6} | {7} | {8} |\n|" \
               "-|---------------------------------------------------------------------------------\n" \
               "|2| {9} | {10} | {11} | {12} | {13} | {14} | {15} | {16} | {17} |\n|" \
               "-|---------------------------------------------------------------------------------\n" \
               "|3| {18} | {19} | {20} | {21} | {22} | {23} | {24} | {25} | {26} |\n|" \
               "-|---------------------------------------------------------------------------------\n" \
               "|4| {27} | {28} | {29} | {30} | {31} | {32} | {33} | {34} | {35} |\n|" \
               "-|---------------------------------------------------------------------------------\n" \
               "|5| {36} | {37} | {38} | {39} | {40} | {41} | {42} | {43} | {44} |\n|" \
               "-|---------------------------------------------------------------------------------\n" \
               "|6| {45} | {46} | {47} | {48} | {49} | {50} | {51} | {52} | {53} |\n|" \
               "-|---------------------------------------------------------------------------------\n" \
               "|7| {54} | {55} | {56} | {57} | {58} | {59} | {60} | {61} | {62} |\n|" \
               "-|---------------------------------------------------------------------------------\n" \
               "|8| {63} | {64} | {65} | {66} | {67} | {68} | {69} | {70} | {71} |\n|" \
               "-|---------------------------------------------------------------------------------\n" \
               "|9| {72} | {73} | {74} | {75} | {76} | {77} | {78} | {79} | {80} |\n|" \
               "-|---------------------------------------------------------------------------------"
NAMES = [' ', 'X', 'O']


class TicTacToe:
    def __init__(self, state=None):
        if state is None:
            state = [0 for i in range(81)]
        self.playerJustMoved = 2
        self.state = state

    def Clone(self):
        state = TicTacToe()
        state.state = self.state[:]
        state.playerJustMoved = self.playerJustMoved
        return state

    def DoMove(self, move):
        assert int(move) >= 0 and int(move) <= 80 and self.state[move] == 0
        self.playerJustMoved = 3 - self.playerJustMoved
        self.state[move] = self.playerJustMoved


    def GetMoves(self):
        if self.checkState() != 0:
            return []

        else:
            moves = []
            for i in range(81):
                if self.state[i] == 0:
                    moves.append(i)

            return moves

    def GetResult(self, playerjm):
        result = self.checkState()
        assert result != 0
        if result == -1:
            return 0.5

        elif result == playerjm:
            return 1.0
        else:
            return 0.0

    def checkState(self):
        for (x, y, z, p, q) in [(0, 1, 2, 3, 4), (9, 10, 11, 12, 13), (18, 19, 20, 21, 22), (27, 28, 29, 30, 31),
                                (36, 37, 38, 39, 40), (45, 46, 47, 48, 49), (54, 55, 56, 57, 58), (63, 64, 65, 66, 67),
                                (1, 2, 3, 4, 5), (10, 11, 12, 13, 14), (19, 20, 21, 22, 23), (28, 29, 30, 31, 32),
                                (37, 38, 39, 40, 41), (46, 47, 48, 49, 50), (55, 56, 57, 58, 59), (64, 65, 66, 67, 68),
                                (2, 3, 4, 5, 6), (11, 12, 13, 14, 15), (20, 21, 22, 23, 24), (29, 30, 31, 32, 33),
                                (38, 39, 40, 41, 42), (47, 48, 49, 50, 51), (56, 57, 58, 59, 60), (65, 66, 67, 68, 69),
                                (3, 4, 5, 6, 7), (12, 13, 14, 15, 16), (21, 22, 23, 24, 25), (30, 31, 32, 33, 34),
                                (39, 40, 41, 42, 43), (48, 49, 50, 51, 52), (57, 58, 59, 60, 61), (66, 67, 68, 69, 70),
                                (4, 5, 6, 7, 8), (13, 14, 15, 16, 17), (22, 23, 24, 25, 26), (31, 32, 33, 34, 35),
                                (40, 41, 42, 43, 44), (49, 50, 51, 52, 53), (58, 59, 60, 61, 62), (67, 68, 69, 70, 71),
                                (0, 9, 18, 27, 36), (1, 10, 19, 28, 37), (2, 11, 20, 29, 38), (3, 12, 21, 30, 39),
                                (4, 13, 22, 31, 40), (5, 14, 23, 32, 41), (6, 15, 24, 33, 42), (7, 16, 25, 34, 43),
                                (8, 17, 26, 35, 44), (9, 18, 27, 36, 45), (10, 19, 28, 37, 46), (11, 20, 29, 38, 47),
                                (12, 21, 30, 39, 48), (13, 22, 31, 40, 49), (14, 23, 32, 41, 50), (15, 24, 33, 42, 51),
                                (16, 25, 34, 43, 52), (17, 26, 35, 44, 53), (18, 27, 36, 45, 54), (19, 28, 37, 46, 55),
                                (20, 29, 38, 47, 56), (21, 30, 39, 48, 57), (22, 31, 40, 49, 58), (23, 32, 41, 50, 59),
                                (24, 33, 42, 51, 60), (25, 34, 43, 52, 61), (26, 35, 44, 53, 62), (27, 36, 45, 54, 63),
                                (28, 37, 46, 55, 64), (29, 38, 47, 56, 65), (30, 39, 48, 57, 66), (31, 40, 49, 58, 67),
                                (32, 41, 50, 59, 68), (33, 42, 51, 60, 69), (34, 43, 52, 61, 70), (35, 44, 53, 62, 71),
                                (36, 45, 54, 63, 72), (37, 46, 55, 64, 73), (38, 47, 56, 65, 74), (39, 48, 57, 66, 75),
                                (40, 49, 58, 67, 76), (41, 50, 59, 68, 77), (42, 51, 60, 69, 78), (43, 52, 61, 70, 79),
                                (44, 53, 62, 71, 80), (0, 10, 20, 30, 40), (1, 11, 21, 31, 41), (2, 12, 22, 32, 42),
                                (3, 13, 23, 33, 43), (4, 14, 24, 34, 44), (9, 19, 29, 39, 49), (10, 20, 30, 40, 50),
                                (11, 21, 31, 41, 51), (12, 22, 32, 42, 52), (13, 23, 33, 43, 53), (18, 28, 38, 48, 58),
                                (19, 29, 39, 49, 59), (20, 30, 40, 50, 60), (21, 31, 41, 51, 61), (22, 32, 42, 52, 62),
                                (27, 37, 47, 57, 67), (28, 38, 48, 58, 68), (29, 39, 49, 59, 69), (30, 40, 50, 60, 70),
                                (31, 41, 51, 61, 71), (36, 46, 56, 66, 76), (37, 47, 57, 67, 77), (38, 48, 58, 68, 78),
                                (39, 49, 59, 69, 79), (40, 50, 60, 70, 80), (4, 12, 20, 28, 36), (5, 13, 21, 29, 37),
                                (6, 14, 22, 30, 38), (7, 15, 23, 31, 39), (8, 16, 24, 32, 40), (5, 13, 21, 29, 37),
                                (10, 18, 26, 34, 42), (11, 19, 27, 35, 43), (12, 20, 28, 36, 44), (13, 21, 29, 37, 45),
                                (14, 22, 30, 38, 46), (11, 19, 27, 35, 43), (12, 20, 28, 36, 44), (13, 21, 29, 37, 45),
                                (18, 26, 34, 42, 50), (19, 27, 35, 43, 51), (20, 28, 36, 44, 52), (21, 29, 37, 45, 53),
                                (22, 30, 38, 46, 54), (23, 31, 39, 47, 55), (24, 32, 40, 48, 56), (25, 33, 41, 49, 57),
                                (26, 34, 42, 50, 58), (23, 31, 39, 47, 55), (24, 32, 40, 48, 56), (25, 33, 41, 49, 57),
                                (30, 38, 46, 54, 62), (31, 39, 47, 55, 63), (32, 40, 48, 56, 64), (33, 41, 49, 57, 65),
                                (34, 42, 50, 58, 66), (31, 39, 47, 55, 63), (32, 40, 48, 56, 64), (33, 41, 49, 57, 65),
                                (38, 46, 54, 62, 70), (39, 47, 55, 63, 71), (40, 48, 56, 64, 72), (41, 49, 57, 65, 73),
                                (42, 50, 58, 66, 74), (39, 47, 55, 63, 71), (40, 48, 56, 64, 72), (41, 49, 57, 65, 73),
                                (43, 51, 59, 67, 75), (44, 52, 60, 68, 76), (1, 9, 10, 11, 19), (2, 10, 11, 12, 20),
                                (3, 11, 12, 13, 21), (4, 12, 13, 14, 22), (5, 13, 14, 15, 23), (6, 14, 15, 16, 24),
                                (7, 15, 16, 17, 25), (10, 18, 19, 20, 28), (11, 19, 20, 21, 29), (12, 20, 21, 22, 30),
                                (13, 21, 22, 23, 31), (14, 22, 23, 24, 32), (15, 23, 24, 25, 33), (16, 24, 25, 26, 34),
                                (19, 27, 28, 29, 37), (20, 28, 29, 30, 38), (21, 29, 30, 31, 39), (22, 30, 31, 32, 40),
                                (23, 31, 32, 33, 41), (24, 32, 33, 34, 42), (25, 33, 34, 35, 43), (28, 36, 37, 38, 46),
                                (29, 37, 38, 39, 47), (30, 38, 39, 40, 48), (31, 39, 40, 41, 49), (32, 40, 41, 42, 50),
                                (33, 41, 42, 43, 51), (34, 42, 43, 44, 52), (37, 45, 46, 47, 55), (38, 46, 47, 48, 56),
                                (39, 47, 48, 49, 57), (40, 48, 49, 50, 58), (41, 49, 50, 51, 59), (42, 50, 51, 52, 60),
                                (43, 51, 52, 53, 61), (46, 54, 55, 56, 64), (47, 55, 56, 57, 65), (48, 56, 57, 58, 66),
                                (49, 57, 58, 59, 67), (50, 58, 59, 60, 68), (51, 59, 60, 61, 69), (52, 60, 61, 62, 70),
                                (55, 63, 64, 65, 73), (56, 64, 65, 66, 74), (57, 65, 66, 67, 75), (58, 66, 67, 68, 76),
                                (59, 67, 68, 69, 77), (60, 68, 69, 70, 78), (61, 69, 70, 71, 79)]:
            if self.state[x] == self.state[y] == self.state[z] == self.state[p] == self.state[q]:
                if self.state[x] == 1:
                    return 1
                elif self.state[x] == 2:
                    return 2

        if [i for i in range(81) if self.state[i] == 0] == []:
            return -1
        return 0

    def __repr__(self):
        cells = []
        for i in range(81):
            cells.append(NAMES[self.state[i]].center(6))
        return BOARD_FORMAT.format(*cells)


class Node:
    def __init__(self, move=None, parent=None, state=None):
        self.move = move
        self.parentNode = parent
        self.childNodes = []
        self.wins = 0
        self.visits = 0
        self.untriedMoves = state.GetMoves()
        self.playerJustMoved = state.playerJustMoved

    def UCTSelectChild(self):
        s = sorted(self.childNodes, key=lambda c: c.wins / c.visits + sqrt(2 * log(self.visits) / c.visits))
        return s[-1]

    def AddChild(self, m, s):
        n = Node(move=m, parent=self, state=copy.deepcopy(s))
        self.untriedMoves.remove(m)
        self.childNodes.append(n)
        return n

    def Update(self, result):
        self.visits += 1
        self.wins += result

    def __repr__(self):
        return "[M" + str(self.move) + " W/V " + str(self.wins) + "/" + str(self.visits) + " U" + str(
            self.untriedMoves) + "]"

    def ChildrenToString(self):
        s = ""
        for c in self.childNodes:
            s += str(c) + "\n"
        return s


def UCT(rootstate, itermax):
    rootnode = Node(state=rootstate)
    sTime = time.time()
    for i in range(itermax):
        node = rootnode
        state = copy.deepcopy(rootstate)

        # selection
        while node.untriedMoves == [] and node.childNodes != []:
            node = node.UCTSelectChild()
            state.DoMove(node.move)

        # Expansion
        if node.untriedMoves != []:
            m = random.choice(node.untriedMoves)
            state.DoMove(m)
            node = node.AddChild(m, state)

        # simulation
        while state.GetMoves() != []:
            state.DoMove(random.choice(state.GetMoves()))

        # BackPropagation
        while node != None:
            node.Update(state.GetResult(node.playerJustMoved))
            node = node.parentNode

    print(rootnode.ChildrenToString())
    eTime = time.time()
    print('AI가 수를 계산하는데 걸린 시간 : ', eTime - sTime)
    s = sorted(rootnode.childNodes, key=lambda c: c.wins / c.visits)
    return sorted(s, key=lambda c: c.visits)[-1].move


def UCTPlayGame():
    state = TicTacToe()
    while state.GetMoves() != []:
        print(str(state))
        if state.playerJustMoved == 1:
            rootstate = copy.deepcopy(state)
            m = UCT(rootstate, itermax=10000)

        else:
            m = input("which Do you want? : ")
            m = int(m)

        state.DoMove(m)

    if state.GetResult(state.playerJustMoved) == 1.0:
        print("Player " + str(state.playerJustMoved) + " Wins!!")

    elif state.GetResult(state.playerJustMoved) == 0.0:
        print("Player " + str(3 - state.playerJustMoved) + " Wins!!")

    else:
        print("Draw!!")


if __name__ == "__main__":
    UCTPlayGame()