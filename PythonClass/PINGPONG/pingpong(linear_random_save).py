from tkinter import *
import random
import time
import numpy as np
import csv
from copy import deepcopy

##########################################################################

############ 공의 위치 파일 저장/불러오기 #############
# 공의 위치 파일 저장(1회만 파일 저장) ####### !!!! 두번째부터는 None으로 놓기!!!! #######
save_ballloc = 'c:\python\data\pingpong_move.csv'
# save_ballloc 파일 위치 입력(반드시 입력해야 함. save_ballloc 의 위치와 동일한 위치로 설정)
load_ballloc = 'c:\python\data\pingpong_move.csv'

############ 회귀분석 가중치 파일 저장/불러오기 #############
# 가중치 파일 저장(저장하고 싶으면 위치 입력. 아니면 None 으로 놓기)
save_weightloc = 'c:\python\data\pingpong_weight.csv'
# save_weightloc 파일 위치 입력(파일 참조하지 않으려면 None 으로 놓기)
load_weightloc = 'c:\python\data\pingpong_weight.csv'

############ 경사감소법 튜닝 ###########
# 경사감소법 learning_rate(변경x)
learning_rate = 0.5
# 경사감소법 시행횟수(변경x)
training_cnt= 10000
#가능조합(learning_rate = 0.00001, training_cnt = 50000)
#가능조합(learning_rate = 0.00002, training_cnt = 25000)

##########################################################################

class Ball:
    def __init__(self, canvas, paddle, color, save=False):

        self.canvas = canvas
        self.paddle = paddle
        self.id = canvas.create_oval(10, 10, 25, 25, fill=color)  # 공 크기 및 색깔
        self.canvas.move(self.id, 245, 100)  # 공을 캔버스 중앙으로 이동
        self.x = random.choice([-4,-3, -2, -1, 1, 2, 3,4])  # 처음 공이 패들에서 움직일때 왼쪽으로 올라갈지 오른쪽으로 올라갈지 랜덤으로 결정되는 부분
        self.y = -3  # 처음 공이 패들에서 움직일때 위로 올라가는 속도
        self.canvas_height = self.canvas.winfo_height()  # 캔버스의 현재 높이를 반환한다.(공이 화면에서 사라지지 않기위해)
        self.canvas_width = self.canvas.winfo_width()  # 캔버스의 현재 넓이를 반환한다.(공이 화면에서 사라지지 않기위해)
        self.hit_bottom = False
        self.save = save
        self.ball_start = []
        self.ball_end = []
        self.convertloc = self.canvas.coords(self.id)[0]
        self.leftorright = 0

    def hit_paddle(self, pos):  # 패들에 공이 튀기게 하는 함수
        paddle_pos = self.canvas.coords(self.paddle.id)
        if self.save == True:
            if pos[3] >= paddle_pos[1] and pos[3] <= paddle_pos[3]:  # 공이 패들에 닿았을때 좌표
                return True
        elif self.save == False:
            if pos[2] >= paddle_pos[0] and pos[0] <= paddle_pos[2]:  # 공이 패들에 내려오기 직전 좌표
                if pos[3] >= paddle_pos[1] and pos[3] <= paddle_pos[3]:  # 공이 패들에 닿았을때 좌표
                    return True
        return False

    ############# 공이 떨어지는 가상의 좌표 ############
    def endloc(self, pos):
        paddle_pos = self.canvas.coords(self.paddle.id)
        if 290 > pos[1] >= 285 and pos[3] <= paddle_pos[3] and self.y > 0:  # 공이 패들 통과할 때의 좌표
            return pos[0]

    def draw(self):
        self.canvas.move(self.id, self.x, self.y)  # 공을 움직이게 하는 부분
        pos = self.canvas.coords(self.id)  # 볼의 현재 좌표를 출력해준다. 공 좌표( 서쪽(0) , 남쪽(1) , 동쪽(2), 북쪽(3) )
        paddle_pos = self.canvas.coords(self.paddle.id)

        #############################################################
        # 가상의 좌표를 만드는 과정
        # self.leftorright는 기본은 0, 최초로 벽에 부딪혔을 때 왼쪽 벽이면 -1, 오른쪽 벽이면 1 을 출력
        if self.leftorright == 0:
            self.convertloc += float(self.x)
        elif self.leftorright != 0:
            self.convertloc += self.leftorright * abs(float(self.x))
        #############################################################
        if pos[1] <= 0:
            self.y *= -1

        if pos[3] >= self.canvas_height:
            self.x = random.choice([-1,1])
            self.y *= -1

        if pos[0] <= 0:
            self.x *= -1
            ######### 최초로 왼쪽 벽에 부딪히면 self.leftorright = -1이 됨 ##########
            if self.leftorright == 0:
                self.leftorright = -1

        if pos[2] >= self.canvas_width:
            self.x *= -1  # 공을 왼쪽으로 돌린다.
            ######### 최초로 오른쪽 벽에 부딪히면 self.leftorright = 1이 됨 ##########
            if self.leftorright == 0:
                self.leftorright = 1

        if self.hit_paddle(pos) == True:
            self.x = random.choice(range(-11,12,2))
            self.y *= -1
            ######### (공의 시작 x좌표, 시작 시 x속력, y속력, 상수1) 을 저장 ##########
            self.ball_start.append([pos[0], float(self.x), float(self.y), 1.0])
            ######### (공이 떨어진 x 좌표) 를 저장
            self.ball_end.append(self.convertloc)
            ######### 패들에 부딪히면, 새로운 공의 시작 정보를 저장하기 위해 가상좌표와 leftorright 값을 초기화 ########
            self.convertloc = pos[0]
            self.leftorright = 0


class Paddle:
    def __init__(self, canvas, color):
        self.canvas = canvas
        self.id = canvas.create_rectangle(0, 0, 100, 10, fill=color)
        self.canvas.move(self.id, 200, 300)
        self.x = 0
        self.canvas_width = self.canvas.winfo_width()

    def draw(self):
        pos = self.canvas.coords(self.id)
        if pos[0] <= 0 and self.x < 0:  # 패들의 위치가 왼쪽 끝이고, 이동하려는 방향이 왼쪽이면 함수 종료(이동 안 함)
            return
        elif pos[2] >= self.canvas_width and self.x > 0:  # 패들의 위치가 오른쪽 끝이고,이동하려는 방향이 오른쪽이면 종료
            return
        self.canvas.move(self.id, self.x, 0)

    ############ 공이 떨어지는 가상의 좌표를 실제 게임 내 좌표로 바꿔주는 메소드 ##############
    def convertendloc(self, convertloc):
        cnt = 0
        if convertloc in range(486):
            return convertloc
        elif convertloc < 0:
            while True:
                if cnt % 2 == 0 and cnt * -485 - convertloc in range(486):
                    return cnt * -485 - convertloc

                elif cnt % 2 == 1 and (cnt + 1) * 485 + convertloc in range(486):
                    return (cnt + 1) * 485 + convertloc
                cnt += 1
        elif convertloc > 485:
            while True:
                if cnt % 2 == 0 and (cnt + 2) * 485 - convertloc in range(486):
                    return (cnt + 2) * 485 - convertloc
                elif cnt % 2 == 1 and (cnt + 1) * -485 + convertloc in range(486):
                    return (cnt + 1) * -485 + convertloc
                cnt += 1

    ############# 회귀분석식을 이용해 공이 떨어질 가상의 위치 예측하는 메소드 ##############
    def prediction(self, input, weight):
        return weight[0] * input[0] + weight[1] * input[1] + weight[2] * input[2] + weight[3] * input[3]

    ############# 공이 떨어질 위치로 패들을 움직이는 메소드 #############
    def predict_move(self, convertloc):
        loc = self.convertendloc(convertloc)
        pos = self.canvas.coords(self.id)
        if pos[0]+40  <loc-5 and pos[2]-40  > loc+10:
            self.x = 0
            print('stop')
        else:
            if pos[2]-40 < loc+10:
                self.x = 3
                print('+3')
            elif pos[0]+40 > loc-5:
                self.x = -3
                print('-3')
        return self.x, 'loc', loc, 'pos', (pos[0],pos[2])

    def move(self, x, y):
        self.x = x

############# 경사감소법 및 회귀분석 머신러닝 ################
class machine_learning():
    ########## 비용함수 메소드 ###########
    @staticmethod
    def Loss(x, y, weight):
        loss = np.sum((x.dot(weight) - y.reshape(len(y),1)) ** 2) / (2 * len(x))
        print(loss)
        return loss

    ########## 경사감소법 및 회귀분석 가중치 계산 메소드 ##########
    @staticmethod
    def gradient_descent(x, alpha=0.00001, descent_cnt=1):
        X = x[:, 0:4]
        Y = x[:, 4]
        M = len(x)
        minloss = 10 ** 20

        WEIGHT = np.zeros((4,1)) # 초기 weight
        loss_history = np.zeros((descent_cnt, 1))

        for cnt in range(descent_cnt):
            predictions = X.dot(WEIGHT).flatten()

            errors_x1 = (predictions - Y) * X[:, 0]
            errors_x2 = (predictions - Y) * X[:, 1]
            errors_x3 = (predictions - Y) * X[:, 2]
            errors_w0 = (predictions - Y) * X[:, 3]

            WEIGHT_backup = deepcopy(WEIGHT)
            # beta = theta - alpha * (X.T.dot(X.dot(beta)-y)/m)
            WEIGHT[0][0] = WEIGHT[0][0] - alpha * (1.0 / M) * errors_x1.sum()
            WEIGHT[1][0] = WEIGHT[1][0] - alpha * (1.0 / M) * errors_x2.sum()
            WEIGHT[2][0] = WEIGHT[2][0] - alpha * (1.0 / M) * errors_x3.sum()
            WEIGHT[3][0] = WEIGHT[3][0] - alpha * (1.0 / M) * errors_w0.sum()

            loss_history[cnt, 0] = machine_learning.Loss(X, Y, WEIGHT)

            ########## BOLD DRIVER 방법 #########
            if minloss >= loss_history[cnt,0]:
                minloss = loss_history[cnt,0]
                alpha *= 1.1
            elif minloss < loss_history[cnt,0]:
                alpha *= 0.5
                WEIGHT = WEIGHT_backup
        return WEIGHT, loss_history


########### 세이브 로드 관련 클래스 ###########
class SaveLoad():
    @staticmethod
    def saveCSV(ballloc, weightloc):
        try:
            if weightloc != None:
                f = open((weightloc), 'a')
                w = csv.writer(f, delimiter=',', lineterminator='\n')

                for key in machine_learning.gradient_descent(np.array(ball_loc_save), learning_rate, training_cnt)[0]:
                    w.writerow(key)
                f.close()
                print('weight saved')
            if ballloc != None:
                f = open((ballloc), 'a')
                w = csv.writer(f, delimiter=',', lineterminator='\n')

                for key in ball_loc_save:
                    w.writerow(key)
                f.close()
                print('ball saved')

        except FileNotFoundError and TypeError:
            print('No Save')

    @staticmethod
    def loadCSV(ballloc,weightloc = None):
        try:
            if weightloc == None :
                pingpong = [data for data in csv.reader(open(ballloc, 'r'))]
                for pp in range(len(pingpong)):
                    for p in range(5):
                        pingpong[pp][p] = float(pingpong[pp][p])
                pingpong = np.array(pingpong)
                return machine_learning.gradient_descent(pingpong,learning_rate, training_cnt)[0]
            else :
                weight = [data for data in csv.reader(open(weightloc, 'r'))]
                return np.array([weight[-4],weight[-3],weight[-2],weight[-1]],dtype=float)

        except FileNotFoundError :
            print('파일 로드 위치를 지정해주세요')


if __name__ == '__main__':

    ############# 머신러닝 위한 시뮬레이션용 ###############
    tk = Tk()  # tk 를 인스턴스화 한다.
    tk.title("Game")  # tk 객체의 title 메소드(함수)로 게임창에 제목을 부여한다.
    tk.resizable(0, 0)  # 게임창의 크기는 가로나 세로로 변경될수 없다라고 말하는것이다.
    tk.wm_attributes("-topmost", 1)  # 다른 모든 창들 앞에 캔버스를 가진 창이 위치할것을 tkinter 에게 알려준다.
    canvas = Canvas(tk, width=500, height=400, bd=0, highlightthickness=0)
    canvas.configure(background='black')

    canvas.pack()  # 앞의 코드에서 전달된 폭과 높이는 매개변수에 따라 크기를 맞추라고 캔버스에에 말해준다.
    tk.update()  # tkinter 에게 게임에서의 애니메이션을 위해 자신을 초기화하라고 알려주는것이다.
    paddle = Paddle(canvas, 'black')
    ball = Ball(canvas, paddle, 'black', save=True)

    for i in range(10000):
        if ball.hit_bottom == False:
            ball.draw()
            paddle.move(paddle.x,0)
            paddle.draw()

    ball_loc_save = []
    for idx_start in range(0,len(ball.ball_start)-1):
        try:
            ball_loc_save.append(ball.ball_start[idx_start]+[ball.ball_end[idx_start+1]])
        except IndexError:
            continue

    ################ 파일 세이브 ################
    SaveLoad.saveCSV(save_ballloc,save_weightloc)

    ################ 파일 로드 ################
    weight = SaveLoad.loadCSV(load_ballloc, load_weightloc)

    ################# 머신러닝 배운 후 플레이 ##################
    paddle = Paddle(canvas, 'white')
    ball = Ball(canvas, paddle, 'white', save=False)

    while True:
        if ball.hit_bottom == False:
            ball.draw()
            try:
                convertloc = int(paddle.prediction(ball.ball_start[-1], weight)[0])
                print('prediction', paddle.predict_move(convertloc))
                paddle.move(paddle.x, 0)
            except IndexError:
                #paddle.move(random.choice([-3, 3]), 0) # 맨처음에 랜덤으로 두게 하려면 활성화
                paddle.move(ball.x,0) # 맨처음에 공을 따라가게 하려면 활성화
            paddle.draw()

        tk.update_idletasks()
        tk.update()
        time.sleep(0.01)
