from tkinter import *
import random
import time
import numpy as np
import csv
from copy import deepcopy
from collections import OrderedDict
from common.layers import *

##########################################################################

############ 공의 위치 파일 저장/불러오기 #############
# 공의 위치 파일 저장(1회만 파일 저장) #######
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
learning_rate = 0.0001
# 경사감소법 시행횟수(변경x)
training_cnt= 30000
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

    def draw(self):
        self.canvas.move(self.id, self.x, self.y)  # 공을 움직이게 하는 부분
        pos = self.canvas.coords(self.id)  # 볼의 현재 좌표를 출력해준다. 공 좌표( 서쪽(0) , 남쪽(1) , 동쪽(2), 북쪽(3) )
        paddle_pos = self.canvas.coords(self.paddle.id)

        if pos[1] <= 0:
            self.y *= -1

        if pos[3] >= self.canvas_height:
            self.x = random.choice([-1,1])
            self.y *= -1

        if pos[0] <= 0:
            self.x *= -1

        if pos[2] >= self.canvas_width:
            self.x *= -1  # 공을 왼쪽으로 돌린다.

        if self.hit_paddle(pos) == True:
            self.x = random.choice(range(-11,12,2))
            self.y *= -1
            ######### (공의 시작 x좌표, 공의 시작 y좌표, 시작 시 x속력, y속력) 을 저장 ##########
            self.ball_start.append([pos[0], pos[1] ,float(self.x), float(self.y)])
            ######### (공이 떨어진 x 좌표) 를 저장
            self.ball_end.append(pos[0])

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

    ############# 회귀분석식을 이용해 공이 떨어질 가상의 위치 예측하는 메소드 ##############
    def prediction(self, input, weight):
        return weight[0] * input[0] + weight[1] * input[1] + weight[2] * input[2] + weight[3] * input[3]

    ############# 공이 떨어질 위치로 패들을 움직이는 메소드 #############
    def predict_move(self, predictedloc):
        loc = predictedloc
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

############# 신경망 ################
class NeuralNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size, hidden_size)
        self.params['b3'] = np.zeros(hidden_size)

        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine3'] = Affine(self.params['W3'], self.params['b3'])

        self.lastlayers = IdentityWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastlayers.forward(y, t)

    def gradient(self, x, t):
        self.loss(x, t)

        dout = self.lastlayers.backward()

        layers = list(self.layers.values())
        layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        grads['W3'] = self.layers['Affine3'].dW
        grads['b3'] = self.layers['Affine3'].db

        return grads


########### 세이브 로드 관련 클래스 ###########
class SaveLoad():
    @staticmethod
    def saveCSV(ballloc, weightloc):
        try:
            # if weightloc != None:
            #     f = open((weightloc), 'a')
            #     w = csv.writer(f, delimiter=',', lineterminator='\n')
            #
            #     for key in NeuralNet.gradient( ):
            #         w.writerow(key)
            #     f.close()
            #     print('weight saved')
            if ballloc != None:
                f = open((ballloc), 'a')
                w = csv.writer(f, delimiter=',', lineterminator='\n')

                for key in ball_loc_save:
                    w.writerow(key)
                f.close()
                print('ball saved')

        except FileNotFoundError and TypeError:
            print('No Save')

    # @staticmethod
    # def loadCSV(ballloc,weightloc = None):
    #     try:
    #         # if weightloc == None :
    #         #     pingpong = [data for data in csv.reader(open(ballloc, 'r'))]
    #         #     for pp in range(len(pingpong)):
    #         #         for p in range(5):
    #         #             pingpong[pp][p] = float(pingpong[pp][p])
    #         #     pingpong = np.array(pingpong)
    #         #     return machine_learning.gradient_descent(pingpong,learning_rate, training_cnt)[0]
    #         else :
    #             weight = [data for data in csv.reader(open(weightloc, 'r'))]
    #             return np.array([weight[-4],weight[-3],weight[-2],weight[-1]],dtype=float)
    #
    #     except FileNotFoundError :
    #         print('파일 로드 위치를 지정해주세요')


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

    for i in range(100000):
        if ball.hit_bottom == False:
            ball.draw()
            paddle.move(paddle.x,0)
            paddle.draw()

    ball_loc_save = []
    for idx_start in range(len(ball.ball_start)-1):
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
