from tkinter import *
import random
import time
import numpy as np
import csv
from copy import deepcopy
class Ball:

    def __init__(self, canvas, paddle, color, save=False):

        self.canvas = canvas
        self.paddle = paddle
        self.id = canvas.create_oval(10, 10, 25, 25, fill=color)  # 공 크기 및 색깔
        self.canvas.move(self.id, 245, 100)  # 공을 캔버스 중앙으로 이동
        self.x = random.choice(range(-3,4))  # 처음 공이 패들에서 움직일때 왼쪽으로 올라갈지 오른쪽으로 올라갈지 랜덤으로 결정되는 부분
        self.y = -3  # 처음 공이 패들에서 움직일때 위로 올라가는 속도
        self.canvas_height = self.canvas.winfo_height()  #캔버스의 현재 높이를 반환한다.(공이 화면에서 사라지지 않기위해)
        self.canvas_width = self.canvas.winfo_width()  # 캔버스의 현재 넓이를 반환한다.(공이 화면에서 사라지지 않기위해)
        self.hit_bottom = False
        self.save = save
        self.ball_start = []
        self.ball_end = []
        self.convertloc = self.canvas.coords(self.id)[0]
        self.leftorright = 0

    def hit_paddle(self, pos):  # 패들에 공이 튀기게 하는 함수
        paddle_pos = self.canvas.coords(self.paddle.id)
        if self.save ==True :
            if pos[3] >= paddle_pos[1] and pos[3] <= paddle_pos[3]:  # 공이 패들에 닿았을때 좌표
                return True
        elif self.save ==False:
            if pos[2] >= paddle_pos[0] and pos[0] <= paddle_pos[2]:  # 공이 패들에 내려오기 직전 좌표
                if pos[3] >= paddle_pos[1] and pos[3] <= paddle_pos[3]:  # 공이 패들에 닿았을때 좌표
                    return True
        return False
    # def startloc(self, pos):
    #     paddle_pos = self.canvas.coords(self.paddle.id)
    #     if 290>pos[1] >=285 and pos[3] <= paddle_pos[3] and self.y < 0:  # 공이 패들 통과할 때의 좌표
    #         return [pos[0], self.x, self.y]

    def endloc(self,pos):
        paddle_pos = self.canvas.coords(self.paddle.id)
        if 290 > pos[1] >= 285 and pos[3] <= paddle_pos[3] and self.y > 0:  # 공이 패들 통과할 때의 좌표
            return pos[0]
    def convertendloc(self,convertlock):
        cnt = 0
        if convertlock in range(486) :
            return convertlock
        elif convertlock <0 :
            while True:
                if cnt % 2 == 0 and cnt * -485 - convertlock in range(486):
                    return cnt * -485 - convertlock

                elif cnt % 2 == 1 and (cnt+1) * 485 + convertlock in range(486):
                    return (cnt+1) * 485 + convertlock
                cnt += 1
        elif convertlock > 485 :
            while True:
                if cnt % 2 == 0 and  (cnt+2) * 485 - convertlock in range(486):
                    return (cnt+2) * 485 - convertlock
                elif cnt % 2 == 1 and (cnt+1) * -485 + convertlock in range(486):
                    return (cnt+1) * -485 + convertlock
                cnt += 1


    def draw(self):

        self.canvas.move(self.id, self.x, self.y)  # 공을 움직이게 하는 부분
        # 공이 화면 밖으로 나가지 않게 해준다
        pos = self.canvas.coords(self.id)  # 볼의 현재 좌표를 출력해준다. 공 좌표( 서쪽(0) , 남쪽(1) , 동쪽(2), 북쪽(3) )
        # [ 255,29,270,44]
        paddle_pos = self.canvas.coords(self.paddle.id)
        if self.leftorright == 0 :
            self.convertloc += float(self.x)
        elif self.leftorright != 0 :
            self.convertloc += self.leftorright * abs(float(self.x))
        # print('convertloc', self.convertloc)
        
        if pos[1] <= 0:  # 공의 남쪽이 가리키는 좌표가 0보다 작아진다면 공이 위쪽 화면 밖으로 나가버리므로

            self.y = 3  # 공을 아래로 떨어뜨린다. (공이 위로 올라갈수로 y 의 값이 작아지므로 아래로 내리려면 다시 양수로)

        if pos[3] >= self.canvas_height:  #공의 북쪽이 가리키는 좌표가 캔버스의 높이보다 더 크다면 화면 아래로 나가버려서

            self.y = -3  # 공을 위로 올린다. (공이 아래로 내려갈수록 y 값이 커지므로 공을 위로 올릴려면 다시 음수로)

        if pos[0] <= 0:  # 공의 서쪽이 가리키는 좌표가 0보다 작으면 공이 화면 왼쪽으로 나가버리므로

            self.x = 3  # 공을 오른쪽으로 돌린다.
            if self.leftorright == 0:
                self.leftorright = -1
                # print('left',self.leftorright)
                # print('pos',pos[0])
                # print('convert',self.convertloc)

        if pos[2] >= self.canvas_width:  # 공의 동쪽이 가리키는 좌표가 공의 넓이보다 크다면 공이 화면 오른쪽으로 나가버림

            self.x = -3  # 공을 왼쪽으로 돌린다.
            if self.leftorright == 0:
                self.leftorright = 1
                # print('left',self.leftorright)
                # print('pos',pos[0])
                # print('convert',self.convertloc)
        # if self.startloc(pos) != None:
        #     self.ball_start.append(self.startloc(pos))

        # if self.endloc(pos) != None:
        #     self.ball_end.append(self.endloc(pos))


        if self.hit_paddle(pos) == True:  # 패들 판에 부딪히면 위로 튕겨올라가게
            self.x = random.choice(range(-3,4))
            self.y = -3  # 공을 위로 올린다.
            self.ball_start.append([pos[0], float(self.x), float(self.y), 1.0])
            self.ball_end.append(self.convertloc)
            # self.ball_end.append(self.convertendloc((self.convertloc)))
            self.convertloc = pos[0]
            self.leftorright = 0

class Paddle:

    def __init__(self, canvas, color):

        self.canvas = canvas
        self.id = canvas.create_rectangle(0, 0, 100, 10, fill=color)  # 패들의 높이와 넓이 그리고 색깔
        self.canvas.move(self.id, 200, 300)  # 패들 사각형을 200,300 에 위치
        self.x = 0  # 패들이 처음 시작할때 움직이지 않게 0으로 설정
        self.canvas_width = self.canvas.winfo_width()  # 캔버스의 넓이를 반환한다. 캔버스 밖으로 패들이 나가지 않도록
        self.canvas.bind_all('<KeyPress-Left>', self.turn_left)  # 왼쪽 화살표 키를 '<KeyPress-Left>'  라는 이름로 바인딩
        self.canvas.bind_all('<KeyPress-Right>', self.turn_right)  # 오른쪽도 마찬가지로 바인딩한다.

    def draw(self):

        pos = self.canvas.coords(self.id)
        #print(pos)

        if pos[0] <= 0 and self.x < 0:  # 패들의 위치가 왼쪽 끝이고, 이동하려는 방향이 왼쪽이면 함수 종료(이동 안 함)

            return

        elif pos[2] >= self.canvas_width and self.x > 0: #패들의 위치가 오른쪽 끝이고,이동하려는 방향이 오른쪽이면 종료

            return

        self.canvas.move(self.id, self.x, 0)



            # 패들이 화면의 끝에 부딪히면 공처럼 튕기는게 아니라 움직임이 멈춰야한다.
            # 그래서 왼쪽 x 좌표(pos[0]) 가 0 과 같거나 작으면 self.x = 0 처럼 x 변수에 0 을
            # 설정한다.  같은 방법으로 오른쪽 x 좌표(pos[2]) 가 캔버스의 폭과 같거나 크면
            # self.x = 0 처럼 변수에 0 을 설정한다.

    def turn_left(self, evt):  # 패들의 방향을 전환하는 함수

        self.x = -3


    def turn_right(self, evt):

        self.x = 3


    def move(self,x,y):

        self.x = x


class machine_learning(object):
    def __init__(self, ball_loc_save, alpha=0.01):
        self.ball_loc_save = ball_loc_save
        self.x = ball_loc_save[:,0:4]
        self.y = ball_loc_save[:,4]
        self.m = len(self.x)
        self.weight = np.zeros((4, 1))
        self.alpha = alpha

    @staticmethod
    def Loss(x, y, weight):
        loss = np.sum((x.dot(weight) - y) ** 2) / (2 * len(x))
        return loss

    @staticmethod
    def gradient_descent(x, alpha=0.0000001, descent_cnt=10000):
        X = x[:, 0:4]
        Y = x[:, 4]
        M = len(x)
        minloss =10**9
        WEIGHT = np.zeros((4,1))
        loss_history = np.zeros((descent_cnt, 1))

        for cnt in range(descent_cnt):
            predictions = X.dot(WEIGHT).flatten()

            errors_x1 = (predictions - Y) * X[:, 0]
            errors_x2 = (predictions - Y) * X[:, 1]
            errors_x3 = (predictions - Y) * X[:, 2]
            errors_w0 = (predictions - Y) * X[:, 3]

            WEIGHT_backup = deepcopy(WEIGHT)
            WEIGHT[0][0] = WEIGHT[0][0] - alpha * (1.0 / M) * errors_x1.sum()
            WEIGHT[1][0] = WEIGHT[1][0] - alpha * (1.0 / M) * errors_x2.sum()
            WEIGHT[2][0] = WEIGHT[2][0] - alpha * (1.0 / M) * errors_x3.sum()
            WEIGHT[3][0] = WEIGHT[3][0] - alpha * (1.0 / M) * errors_w0.sum()

            # for idx in range(M):
            #     dtheta = np.sum((X.dot(WEIGHT) - Y) * X[:, idx]) / len(x)
            #     WEIGHT[idx][0] = WEIGHT[idx][0] - alpha * dtheta
            loss_history[cnt, 0] = machine_learning.Loss(X, Y, WEIGHT)

            if minloss >= loss_history[cnt,0]:
                minloss = loss_history[cnt,0]
                alpha *= 1.1
            elif minloss < loss_history[cnt,0]:
                alpha *= 0.5
                WEIGHT = WEIGHT_backup
        return WEIGHT, loss_history, minloss

def saveCSV(loc='d:\python\data\pingpong.csv'):
    if not loc is None:
        f = open((loc), 'a')
        w = csv.writer(f, delimiter=',', lineterminator='\n')

        for key in ball_loc_save:
            w.writerow(key)
        f.close()

def normalize(x):
    n = len(x[0])
    for i in range(n):
        mean = np.mean(x[:, i])
        std = np.std(x[:, i])
        if std ==0:
            x[:, i] = 0
        else:
            x[:, i] = (x[:, i] - mean) / std
    return x
#
# def readCSV(loc ='d:\python\data\pingpong.csv' ):  # 파일로 저장된 가중치값을 읽어들일 때 사용하는 메소드
#     if loc != 'None':
#         try:
#             file = open(loc, 'r')  # 파일은 __main__절에서 다음 형식으로 저장됨.
#
#             values_list = csv.reader(file)
#             return values_list
#         except FileNotFoundError:
#             return None

tk = Tk()  # tk 를 인스턴스화 한다.

tk.title("Game")  # tk 객체의 title 메소드(함수)로 게임창에 제목을 부여한다.

tk.resizable(0, 0)  # 게임창의 크기는 가로나 세로로 변경될수 없다라고 말하는것이다.

tk.wm_attributes("-topmost", 1)  # 다른 모든 창들 앞에 캔버스를 가진 창이 위치할것을 tkinter 에게 알려준다.

canvas = Canvas(tk, width=500, height=400, bd=0, highlightthickness=0)

# bg=0,highlightthickness=0 은 캔버스 외곽에 둘러싼

# 외곽선이 없도록 하는것이다. (게임화면이 좀더 좋게)


canvas.pack()  # 앞의 코드에서 전달된 폭과 높이는 매개변수에 따라 크기를 맞추라고 캔버스에에 말해준다.

tk.update()  # tkinter 에게 게임에서의 애니메이션을 위해 자신을 초기화하라고 알려주는것이다.

paddle = Paddle(canvas, 'blue')

ball = Ball(canvas, paddle, 'red', save=True)

start = False

# 공을 약간 움직이고 새로운 위치로 화면을 다시 그리며, 잠깐 잠들었다가 다시 시작해 ! "

for i in range(1):

    if ball.hit_bottom == False:

        ball.draw()

        paddle.move(paddle.x,0)

        paddle.draw()
        # print('start', len(ball.ball_start))
        # print(ball.ball_start)
        # print('end', len(ball.ball_end))
        # print(ball.ball_end)
    tk.update_idletasks()  # 우리가 창을 닫으라고 할때까지 계속해서 tkinter 에게 화면을 그려라 !

    tk.update()  # tkinter 에게 게임에서의 애니메이션을 위해 자신을 초기화하라고 알려주는것이다.

    # time.sleep(0.01)  # 무한 루프중에 100분의 1초마다 잠들어라 !
ball_loc_save = []

for idx_start in range(0,len(ball.ball_start)-1):
    try:
        ball_loc_save.append(ball.ball_start[idx_start]+[ball.ball_end[idx_start+1]])
    except IndexError:
        continue
# print(ball_loc_save)
# saveCSV()

pingpong = [data for data in csv.reader(open('d:\python\data\pingpong.csv', 'r'))]
for pp in range(len(pingpong)):
    for p in range(5):
        pingpong[pp][p] = float(pingpong[pp][p])
pingpong= np.array(pingpong)
weight = machine_learning.gradient_descent(pingpong)[0]
print(weight)

def prediction(a,b,c,d,weight):
    return weight[0]*a +weight[1]*b +weight[2] *c + weight[3] *d
print(prediction(388,2,-3,1,weight))
print(prediction(57,1,-3,1,weight))
print(prediction(249,2,-3,1,weight))

