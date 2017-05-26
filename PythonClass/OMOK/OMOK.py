<<<<<<< HEAD
# 사람 대 사람 오목 게임


EMPTY = 0  # 비어있는 칸은 0으로
DRAW = 3  # 비긴 경우는 3으로

# 9x9 바둑판 만들기

BOARD_FORMAT = "----1----2----3----4----5----6----7----8----9--\n1| {0} | {1} | {2} | {3} | {4} | {5} | {6} | {7} | {8} |\n|" \
               "-----------------------------------------------\n2| {9} | {10} | {11} | {12} | {13} | {14} | {15} | {16} | {17} |\n|" \
               "-----------------------------------------------\n3| {18} | {19} | {20} | {21} | {22} | {23} | {24} | {25} | {26} |\n|" \
               "-----------------------------------------------\n4| {27} | {28} | {29} | {30} | {31} | {32} | {33} | {34} | {35} |\n|" \
               "-----------------------------------------------\n5| {36} | {37} | {38} | {39} | {40} | {41} | {42} | {43} | {44} |\n|" \
               "-----------------------------------------------\n6| {45} | {46} | {47} | {48} | {49} | {50} | {51} | {52} | {53} |\n|" \
               "-----------------------------------------------\n7| {54} | {55} | {56} | {57} | {58} | {59} | {60} | {61} | {62} |\n|" \
               "-----------------------------------------------\n8| {63} | {64} | {65} | {66} | {67} | {68} | {69} | {70} | {71} |\n|" \
               "-----------------------------------------------\n9| {72} | {73} | {74} | {75} | {76} | {77} | {78} | {79} | {80} |\n" \
               "------------------------------------------------"
# Names[0] -> 비어있는 경우 ' ', Names[1] -> Player1 의 돌은 흑돌, Names[2] -> Player2 의 돌은 백돌
Names = ['  ', '●', '○']


# 바둑판에 돌 놓기
def printboard(state):
    ball = []  # 임의의 리스트 ball 을 만들어서
    for i in range(9):
        for j in range(9):
            ball.append(Names[state[i][j]])  # 바둑판 좌표 state[i][j] 의 값이 1이면 Names[1] -> 흑돌을 리스트에 입력
    print(BOARD_FORMAT.format(*ball))  # 처음에 만든 BOARD_FORMAT 바둑판에 돌 놓기


# 초기 바둑판 좌표값
def emptyboard():
    empty_board = [[EMPTY for i in range(9)] for i in range(9)]
    return empty_board  # [ [EMPTY,EMPTY,EMPTY... EMPTY 총 9개], [EMPTY,EMPTY,EMPTY... EMPTY 총 9개], ...]
    # [ [EMPTY x 9] x 9 ]  로 이뤄진 리스트 만들기. 즉 모든 값이 EMPTY 인 9x9의 리스트 생성


# 게임이 종료되는 조건 함수 생성
def gameover(state):
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
                if state[i][j] * state[i + 1][j + 1] * state[i + 2][j + 2] * state[i + 3][j + 3] * state[i + 4][
                            j + 4] == 1:
                    return 1
                if state[i][j + 4] * state[i + 1][j + 3] * state[i + 2][j + 2] * state[i + 3][j + 1] * state[i + 4][
                    j] == 1:
                    return 1
                    # 대각선으로 다섯칸 모두 2인 경우(player2의 백돌이 대각선으로 연속 다섯칸에 놓인 경우)
                if state[i][j] * state[i + 1][j + 1] * state[i + 2][j + 2] * state[i + 3][j + 3] * state[i + 4][
                            j + 4] == 32:
                    return 2
                if state[i][j + 4] * state[i + 1][j + 3] * state[i + 2][j + 2] * state[i + 3][j + 1] * state[i + 4][
                    j] == 32:
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


class human():
    def __init__(self, player):
        self.player = player

    # 돌 놓기
    def Action(self, state):
        printboard(state)  # 바둑판 출력
        action = None
        switch_map = {}  # 바둑판 좌표 딕셔너리
        for i in range(1, 10):
            for j in range(1, 10):
                switch_map[10 * i + j] = (i, j)

                # 인풋 받기
        while action not in range(11, 100) or state[switch_map[action][0] - 1][switch_map[action][1] - 1] != EMPTY:
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


# 게임 진행
def play(p1, p2):
    state = emptyboard()
    for i in range(81):
        if i % 2 == 0:
            move = p1.Action(state)
        else:
            move = p2.Action(state)
        state[move[0] - 1][move[1] - 1] = i % 2 + 1
        winner = gameover(state)
        if winner != EMPTY:
            printboard(state)
            return winner
    return winner


if __name__ == '__main__':
    p1 = human(1)
    p2 = human(2)
    while True:
        winner = play(p1, p2)
        p1.episode_over(winner)
        if winner != '':
=======
# 사람 대 사람 오목 게임


EMPTY = 0  # 비어있는 칸은 0으로
DRAW = 3  # 비긴 경우는 3으로

# 9x9 바둑판 만들기

BOARD_FORMAT = "----1----2----3----4----5----6----7----8----9--\n1| {0} | {1} | {2} | {3} | {4} | {5} | {6} | {7} | {8} |\n|" \
               "-----------------------------------------------\n2| {9} | {10} | {11} | {12} | {13} | {14} | {15} | {16} | {17} |\n|" \
               "-----------------------------------------------\n3| {18} | {19} | {20} | {21} | {22} | {23} | {24} | {25} | {26} |\n|" \
               "-----------------------------------------------\n4| {27} | {28} | {29} | {30} | {31} | {32} | {33} | {34} | {35} |\n|" \
               "-----------------------------------------------\n5| {36} | {37} | {38} | {39} | {40} | {41} | {42} | {43} | {44} |\n|" \
               "-----------------------------------------------\n6| {45} | {46} | {47} | {48} | {49} | {50} | {51} | {52} | {53} |\n|" \
               "-----------------------------------------------\n7| {54} | {55} | {56} | {57} | {58} | {59} | {60} | {61} | {62} |\n|" \
               "-----------------------------------------------\n8| {63} | {64} | {65} | {66} | {67} | {68} | {69} | {70} | {71} |\n|" \
               "-----------------------------------------------\n9| {72} | {73} | {74} | {75} | {76} | {77} | {78} | {79} | {80} |\n" \
               "------------------------------------------------"
# Names[0] -> 비어있는 경우 ' ', Names[1] -> Player1 의 돌은 흑돌, Names[2] -> Player2 의 돌은 백돌
Names = ['  ', '●', '○']


# 바둑판에 돌 놓기
def printboard(state):
    ball = []  # 임의의 리스트 ball 을 만들어서
    for i in range(9):
        for j in range(9):
            ball.append(Names[state[i][j]])  # 바둑판 좌표 state[i][j] 의 값이 1이면 Names[1] -> 흑돌을 리스트에 입력
    print(BOARD_FORMAT.format(*ball))  # 처음에 만든 BOARD_FORMAT 바둑판에 돌 놓기


# 초기 바둑판 좌표값
def emptyboard():
    empty_board = [[EMPTY for i in range(9)] for i in range(9)]
    return empty_board  # [ [EMPTY,EMPTY,EMPTY... EMPTY 총 9개], [EMPTY,EMPTY,EMPTY... EMPTY 총 9개], ...]
    # [ [EMPTY x 9] x 9 ]  로 이뤄진 리스트 만들기. 즉 모든 값이 EMPTY 인 9x9의 리스트 생성


# 게임이 종료되는 조건 함수 생성
def gameover(state):
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
                if state[i][j] * state[i + 1][j + 1] * state[i + 2][j + 2] * state[i + 3][j + 3] * state[i + 4][
                            j + 4] == 1:
                    return 1
                if state[i][j + 4] * state[i + 1][j + 3] * state[i + 2][j + 2] * state[i + 3][j + 1] * state[i + 4][
                    j] == 1:
                    return 1
                    # 대각선으로 다섯칸 모두 2인 경우(player2의 백돌이 대각선으로 연속 다섯칸에 놓인 경우)
                if state[i][j] * state[i + 1][j + 1] * state[i + 2][j + 2] * state[i + 3][j + 3] * state[i + 4][
                            j + 4] == 32:
                    return 2
                if state[i][j + 4] * state[i + 1][j + 3] * state[i + 2][j + 2] * state[i + 3][j + 1] * state[i + 4][
                    j] == 32:
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


class human():
    def __init__(self, player):
        self.player = player

    # 돌 놓기
    def Action(self, state):
        printboard(state)  # 바둑판 출력
        action = None
        switch_map = {}  # 바둑판 좌표 딕셔너리
        for i in range(1, 10):
            for j in range(1, 10):
                switch_map[10 * i + j] = (i, j)

                # 인풋 받기
        while action not in range(11, 100) or state[switch_map[action][0] - 1][switch_map[action][1] - 1] != EMPTY:
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


# 게임 진행
def play(p1, p2):
    state = emptyboard()
    for i in range(81):
        if i % 2 == 0:
            move = p1.Action(state)
        else:
            move = p2.Action(state)
        state[move[0] - 1][move[1] - 1] = i % 2 + 1
        winner = gameover(state)
        if winner != EMPTY:
            printboard(state)
            return winner
    return winner


if __name__ == '__main__':
    p1 = human(1)
    p2 = human(2)
    while True:
        winner = play(p1, p2)
        p1.episode_over(winner)
        if winner != '':
>>>>>>> init
            break  # winner 가 있을 경우 루프 벗어남