from tkinter import *
import random
import csv
import time
import math

MAN_MOVE = [-6, 0, 6]
# RANDOM_POS = [6*i for i in range(98)]
# random.shuffle(RANDOM_POS)
# print(RANDOM_POS)
RANDOM_POS = [558, 486, 444, 282, 54, 516, 108, 348, 480, 120, 222, 390, 468, 426, 450, 570, 432, 246, 60, 414, 342, 12, 384, 522, 456, 288, 30, 492, 162, 78, 420, 546, 336, 84, 132, 402, 582, 210, 498, 102, 144, 258, 180, 90, 36, 126, 24, 264, 540, 204, 192, 156, 216, 534, 252, 186, 6, 138, 48, 198, 42, 318, 294, 366, 174, 300, 576, 354, 378, 72, 168, 408, 324, 0, 510, 96, 504, 234, 330, 372, 360, 240, 66, 396, 150, 312, 474, 552, 564, 438, 228, 270, 462, 114, 306, 528, 276, 18]
# RANDOM_POS = [396, 255, 561, 72, 171, 219, 117, 24, 495, 375, 231, 342, 321, 99, 336, 519, 123, 258, 135, 291, 3, 180, 588, 483, 381, 246, 489, 333, 552, 30, 63, 465, 480, 174, 414, 183, 318, 27, 546, 429, 108, 153, 363, 438, 504, 249, 402, 447, 384, 18, 126, 510, 222, 177, 324, 84, 555, 87, 285, 522, 204, 240, 69, 300, 492, 144, 15, 537, 327, 570, 129, 90, 165, 441, 411, 60, 528, 435, 294, 33, 567, 243, 534, 390, 474, 543, 450, 387, 462, 6, 456, 45, 372, 42, 444, 141, 405, 207, 513, 66, 225, 168, 114, 297, 408, 195, 48, 264, 288, 198, 228, 369, 39, 564, 471, 150, 507, 348, 360, 351, 201, 339, 282, 234, 270, 426, 477, 147, 273, 81, 96, 573, 366, 261, 159, 345, 306, 78, 252, 303, 357, 558, 237, 420, 540, 417, 132, 57, 9, 162, 309, 186, 102, 330, 393, 111, 378, 192, 432, 501, 582, 486, 105, 459, 423, 279, 516, 312, 453, 36, 189, 576, 549, 216, 213, 156, 210, 354, 579, 468, 120, 531, 51, 21, 525, 54, 585, 93, 12, 399, 315, 138, 276, 75, 498, 267, 0]
KEY_VALUE = {}   # 좌표를 key으로, 가중치(Q)를 value로 갖는 딕셔너리

# RANDOM_POS = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96, 99, 102, 105, 108, 111, 114, 117, 120, 123, 126, 129, 132, 135, 138, 141, 144, 147, 150, 153, 156, 159, 162, 165, 168, 171, 174, 177, 180, 183, 186, 189, 192, 195, 198, 201, 204, 207, 210, 213, 216, 219, 222, 225, 228, 231, 234, 237, 240, 243, 246, 249, 252, 255, 258, 261, 264, 267, 270, 273, 276, 279, 282, 285, 288, 291, 294, 297, 300, 303, 306, 309, 312, 315, 318, 321, 324, 327, 330, 333, 336, 339, 342, 345, 348, 351, 354, 357, 360, 363, 366, 369, 372, 375, 378, 381, 384, 387, 390, 393, 396, 399, 402, 405, 408, 411, 414, 417, 420, 423, 426, 429, 432, 435, 438, 441, 444, 447, 450, 453, 456, 459, 462, 465, 468, 471, 474, 477, 480, 483, 486, 489, 492, 495, 498, 501, 504, 507, 510, 513, 516, 519, 522, 525, 528, 531, 534, 537, 540, 543, 546, 549, 552, 555, 558, 561, 564, 567, 570, 573, 576, 579, 582, 585, 588]






############################################################################################여기는 사람
# class Man_me:
#     def __init__(self, canvas):
#         self.canvas = canvas
#         self.man = canvas.create_rectangle(0, 0, 10, 20, fill='magenta')
#         self.canvas.move(self.man, 295, 480)
#         self.x = 0
#         self.y = 0
#         # self.canvas_width = self.canvas.winfo_width()      #canvas 가로
#         # self.canvas_height = self.canvas.winfo_height()    #canvas 세로
#         self.canvas.bind_all('<KeyPress-Left>', self.turn_left)
#         self.canvas.bind_all('<KeyPress-Right>', self.turn_right)   #canvas가 감지
#         self.canvas.bind_all('<KeyPress-Down>', self.stop)
#         # print(self.canvas_width)
#
#     def draw(self):
#         man_pos = self.canvas.coords(self.man)       #self.man의 좌상우하의 좌표, 위치
#
#         if man_pos[0] <= 0 and self.x < 0:                      #self.man이 오른쪽으로 나가지 않도록
#             self.x = 5
#         elif man_pos[2] >= 600 and self.x > 0:
#             self.x = -5
#
#         self.canvas.move(self.man, self.x, self.y)
#
#
#     def turn_left(self, evt):
#         self.x = -5
#
#
#     def turn_right(self, evt):
#         self.x = 5
#
#     def stop(self, evt):
#         self.x = 0
############################################################################################여기는 사람


class Man_AI:
    DODGEPNT = 0

    def __init__(self, canvas):
        self.canvas = canvas
        self.man = canvas.create_rectangle(0, 0, 10, 20, fill='magenta')
        self.canvas.move(self.man, 295, 480)         # self.man 그리기
        self.x = 0
        self.y = 0
        self.man_pos = self.canvas.coords(self.man)  # self.man의 좌상우하의 좌표, 위치 선언


    def draw(self):
        self.man_pos = self.canvas.coords(self.man)

        # self.man이 화면 밖으로 나가지 않도록 하는 코드
        if self.man_pos[0] <= 0 and self.x < 0:
            self.x = 6
        elif self.man_pos[2] >= 600 and self.x > 0:
            self.x = -6

        # random()에서 받은 속력값을 대입해서 self.man 그리기
        self.canvas.move(self.man, self.x, self.y)


    def move(self, x):
        # 랜덤으로 self.man이 움직일 속력 반환
        self.x = x




class Poop:
    CYCLE_DATA = []
    HIT = 0

    def __init__(self, canvas, man, rdm_pos):   # append 되면서 실행
        self.get_point = 0   # 득점상황
        self.poop_x = 0      # self.poop의 x좌표
        self.man_x = 0       # self.man의 x좌표
        self.man_speed = 0   # self.man의 속력

        self.canvas = canvas
        self.man = man
        self.poop = canvas.create_oval(0, 0, 10, 10, fill='#E86A0C')
        # self.poop = canvas.create_oval(0, 0, 10, 10, fill='pink')
        self.x = 0
        self.y = 6   # poop이 떨어지는 속도 조절
        self.random_pos = None
        self.learning = True
        self.alpha = 0.99   # 망각계수
        self.rdm_pos = rdm_pos
        self.random()
        self.canvas.move(self.poop, self.x_pos, 0)      # self.poop 그리기
        self.poop_pos = self.canvas.coords(self.poop)   # self.poop의 좌상우하의 좌표, 위치 선언


    def random(self):  # 학습 결과 여기서 바꿔주면 됩니다.
        self.x_pos = self.rdm_pos


    def action(self):
        self.canvas.move(self.poop, self.x, self.y)   # 똥 그리기, self.poop_pos도 같이 update함
        direction = self.greedy_choice()   # 처음에는 judgement_point()에 걸리지 않기 때문에 None을 반환한다.
        x = MAN_MOVE[direction]   # greedy_choice()에서 가장 적절한 items 번호 반환해서 사용

        if self.judgement_point():   # 여기서 결정된 행동이 기록된다.
            Poop.CYCLE_DATA.append(self.keystate(direction))   # greedy_choice에서 나온 값을 적용해서 append, 강화학습위한 데이터 저장
            self.Qmaker()

        self.man.move(x)


    def keystate(self, man_speed):   # 여기서 점수들을 판정해주고 판정당시의 정보들을 모아준다.
        if self.judgement_point():
            self.man_x = int(self.man.man_pos[0])
            self.man_speed = self.man.x
            self.poop_x = int(self.poop_pos[0])
            return (self.man_x, self.poop_x, man_speed)


    def judgement_point(self):   # 일정 높이에 도달했는지 판정
        self.poop_pos = self.canvas.coords(self.poop)

        # 득점 상황 선언
        if self.man.man_pos[1] <= self.poop_pos[3] <= self.man.man_pos[3]-11:
            return True
        return False


    def hit_man(self):   # 일정 높이에 도달했을 때 맞았는지 안맞았는지 반환
        if self.man.man_pos[0] <= self.poop_pos[0] <= self.man.man_pos[2]\
                or self.man.man_pos[0] <= self.poop_pos[2] <= self.man.man_pos[2]:
            return True
        return False


    # Q를 계산해서 가장 적절한 값을 선택한다.
    def greedy_choice(self):   # 똥 하나에 세개의 값을 반환해서 KEY_VALUE에 쌓는다.
        val_left = self.keystate(0)
        val_stop = self.keystate(1)
        val_right = self.keystate(2)

        # Q 비교
        if self.lookup(val_left) > self.lookup(val_stop)\
            and self.lookup(val_left) > self.lookup(val_right):
            return 0
        elif self.lookup(val_stop) > self.lookup(val_left)\
            and self.lookup(val_stop) > self.lookup(val_right):
            return 1
        elif self.lookup(val_right) > self.lookup(val_left)\
            and self.lookup(val_right) > self.lookup(val_stop):
            return 2
        else:   # 적당한 값이 없을 경우, 즉 모두 0이거나 수가 같아서 비교가 불가능할 경우
            return random.choice([0, 1, 2])


    # 새로 들어온 key에 대한 value 지정 및 추가(add)
    def add(self, key):
        KEY_VALUE[key] = 0


    # serf.values에 해당 key가 없으면.... 인데 어디서 추가하는거지?
    def lookup(self, key):
        if key not in KEY_VALUE:
            self.add(key)
        return KEY_VALUE[key]


    def reset_condition(self):
        if self.judgement_point():
            if self.hit_man():
                return True
        return False


    def reset(self):
        Poop.CYCLE_DATA = []   # 게임 끝나면 다시 Q값을 구해주기 위해 비워준다. for문에서 드르륵 여러번 선언, 상관없음
        # self.y = 1000


    def reinforcement(self, newVal, idx):
        if idx >= 0 and self.learning:
            preVal = round(KEY_VALUE[Poop.CYCLE_DATA[idx]], 5)   # 점수판정 바로 전의 데이터에 대한 Q값
            KEY_VALUE[Poop.CYCLE_DATA[idx]] += round((self.alpha * (newVal - preVal)), 5)   # 결과가 일어나기 직전의 선택부터 차례대로 뒤로 가면서 가중치를 적용, 역전파 적용
            # print(KEY_VALUE[Poop.CYCLE_DATA[idx]])
            return self.reinforcement(newVal * self.alpha, idx - 1)   # 재귀


    def punishment(self, newVal, idx):
        if idx >= len(Poop.CYCLE_DATA)-4 and self.learning:
            preVal = round(KEY_VALUE[Poop.CYCLE_DATA[idx]], 5)   # 점수판정 바로 전의 데이터에 대한 Q값
            KEY_VALUE[Poop.CYCLE_DATA[idx]] += round((self.alpha * (newVal - preVal)), 5)   # 결과가 일어나기 직전의 선택부터 차례대로 뒤로 가면서 가중치를 적용, 역전파 적용
            return self.punishment(newVal * self.alpha, idx-1)  # 재귀


    def Qmaker(self):
        if rotation_cnt % 1000 == 0:  # save_term, csv파일에 담는 주기
            self.writeCSV()

        if self.hit_man():
            Poop.HIT += 1
            result_value = -1   # 실패에 대한 보상치
            return self.punishment(result_value, len(Poop.CYCLE_DATA) - 2)

        else:
            result_value = 1    # 성공에 대한 보상치
            Man_AI.DODGEPNT += 1
            return self.reinforcement(result_value, len(Poop.CYCLE_DATA) - 2)




    def check_out(self):
        self.poop_pos = self.canvas.coords(self.poop)
        if self.poop_pos[3] >= 520:
            return True
        return False


    def writeCSV(self):
        Fn = open("c:\\data\\poop_val.csv", 'w', newline='')
        writer = csv.writer(Fn, delimiter=',')   # 구분자 comma(csv파일 저장)
        writer.writerow([rotation_cnt])   # self.poop 한바퀴 돌때마다 저장
        keys = KEY_VALUE.keys()
        # print(keys)
        for key in keys:
            try:
                writer.writerow([
                                    key[0],
                                    key[1],
                                    key[2],
                                    KEY_VALUE[key]
                                    ])
            except:
                pass
        Fn.close()


    def loadCSV(self):
        try:
            Fn = open("c:\\data\\poop_val.csv", 'r')
            rotation_cnt = int(Fn.readline().split(',')[0])  # 첫 줄의 학습 게임 횟수 불러오기
            reader = csv.reader(Fn, delimiter=',')
            for key in reader:
                KEY_VALUE[(
                    int(float(key[0])), int(float(key[1])), int(float(key[2])),
                    )] = float(key[3])
            print('Load Success! Start at cycle {0}'.format(rotation_cnt))
        except Exception:
            print('Load Failed!')


    def __del__(self):
        return 'del'




tk = Tk()
tk.title("Dodge Your Poop Faster")   #게임 창의 제목 출력
tk.resizable(0, 0)                   #tk.resizable(가로크기조절, 세로크기조절)
tk.wm_attributes("-topmost", 1)      #생성된 게임창을 다른창의 제일 위에 오도록 정렬
tk.update()  # 여기서 한번 다시 적어준다.


canvas = Canvas(tk, width=600, height=500, bd=0, highlightthickness=0)
#bd=0, highlightthickness=0 은 베젤의 크기를 의미한다.
canvas.configure(background='#E8D487')
canvas.pack()  #앞의 코드에서 전달된 폭과 높이는 매개변수에 따라 크기를 맞추라고 캔버스에에 말해준다.


man = Man_AI(canvas)
poop = []


rotation_cnt = 0
while 1:
    rotation_cnt += 1

    for rdm_pos in RANDOM_POS:

        tk.update()
        tk.update_idletasks()
        poop.append(Poop(canvas, man, rdm_pos))   # 객체 생성, Poop.POOP_X
        man.draw()

        for i in range(len(poop)):  # poop에 들어가있는 객체의 마지막 순서에 해당하는 객체의 메소드가 실행된다!
            try:
                poop[i].action()   # 객체 실행, 일부에서 judgement_point()가 걸린다.
                if poop[i].reset_condition():   # 전체에서 judgement_point()과 hit_man()이 걸린다.
                    poop[i].reset()
            except IndexError:
                continue

            if poop[i].check_out():
                del poop[i]

    # if game_cnt >= 3500:
    #     time.sleep(0.05)

    if rotation_cnt % 15 == 0:
        print('맞은횟수 :', Poop.HIT)   # Poop.judgement_point에 걸리지 않은 self.poop 고려
        Poop.HIT = 0
        print('rotation횟수 :', rotation_cnt)
        print('KEY_VALUE 데이터 개수 : {}/29590'.format(len(KEY_VALUE)))
        print('===============================')


    if rotation_cnt % 195 == 0:
        print(KEY_VALUE)
