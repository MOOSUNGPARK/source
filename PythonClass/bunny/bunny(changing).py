# 1 - Import library
import math
import random
import pygame
from pygame.locals import *
import numpy as np
from collections import deque

def rgb2gray(rgb):
    '''
        YCrCb : 디지털(CRT, LCDl, PDP 등)을 위해서 따로 만들어둔 표현방법.
         - Y = Red*0.2126 + Green*0.7152 + Blue*0.0722
        YPbPr : 아날로그 시스템을 위한 표현방법.
         - Y : Red*0.299 + Green*0.587 + Blue*0.114
        실제 RGB 값들을 Gray Scale 로 변환하는 함수 .
    '''
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2126 * r + 0.7152 * g + 0.0722 * b

    return np.array(gray).astype('int32')

def rebin(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)

# 2 - Initialize the game
pygame.init()
pygame.mixer.init()
width, height = 640, 480
screen = pygame.display.set_mode((width, height))
keys = [False, False]
playerpos = [100, 100]
acc = [0, 0]
arrows = []
badtimer = 100
badtimer1 = 0
badguys = [[640, 100]]
healthvalue = 194
state = deque(maxlen=2)

# 3 - Load images
player = pygame.image.load("resources/images/dude.png")
grass = pygame.image.load("resources/images/grass.png")
castle = pygame.image.load("resources/images/castle.png")
arrow = pygame.image.load("resources/images/bullet.png")
badguyimg1 = pygame.image.load("resources/images/badguy.png")
badguyimg = badguyimg1
healthbar = pygame.image.load("resources/images/healthbar.png")
health = pygame.image.load("resources/images/health.png")
gameover = pygame.image.load("resources/images/gameover.png")
youwin = pygame.image.load("resources/images/youwin.png")

# 3.1 - Load audio
hit = pygame.mixer.Sound("resources/audio/explode.wav")
enemy = pygame.mixer.Sound("resources/audio/enemy.wav")
shoot = pygame.mixer.Sound("resources/audio/shoot.wav")
hit.set_volume(0.05)
enemy.set_volume(0.05)
shoot.set_volume(0.05)
pygame.mixer.music.load('resources/audio/moonlight.wav')
pygame.mixer.music.play(-1, 0.0)
pygame.mixer.music.set_volume(0.25)

# 4 - keep looping through
# 4 - keep looping through
# running = 1
# exitcode = 0

while True :
    running = 1
    exitcode = 0
    screen = pygame.display.set_mode((width, height))
    keys = [False, False, False, False]
    playerpos = [100, 100]
    acc = [0, 0]
    arrows = []
    badtimer = 100
    badtimer1 = 0
    badguys = [[640, 100]]
    healthvalue = 194
    score = 0


    while running:
        # 5 - clear the screen before drawing it again
        screen.fill(0)
        # print(pygame.surfarray.array3d(screen).shape)


        # with open("c:\\python\\data\\bunny.csv", "a", encoding="utf-8") as f:
        #     for row in surf :
        #         print([str(row[i])+'\n'+',' for i in range(len(row))])


        # font = pygame.font.Font(None, 24)
        # Acc = font.render("Accuracy: "+str(acc[0] * 1.0 / (acc[1] * 100 + 1e-7))+"%", True, (0, 0, 0))
        # textRect1 = Acc.get_rect()
        # textRect1.topright = [635,5]
        # screen.blit(Acc, textRect1)


        # 6 - draw the screen elements
        for x in range(width//grass.get_width()+1):
            for y in range(height//grass.get_height()+1):
                screen.blit(grass, (x*100, y*100))
        screen.blit(castle, (0, 30))
        screen.blit(castle, (0, 135))
        screen.blit(castle, (0, 240))
        screen.blit(castle, (0, 345))

        # 6.1 - Set player position and rotation
        position = pygame.mouse.get_pos()
        # angle = math.atan2(position[1]-(playerpos[1]+32),position[0]-(playerpos[0]+26))
        playerpos1 = (playerpos[0]-player.get_rect().width/2, playerpos[1]-player.get_rect().height/2)
        screen.blit(player, playerpos1)

        # 6.2 - Draw arrows
        for bullet in arrows:
            # print(bullet)
            index = 0
            # velx = math.cos(bullet[0])*10
            # print(velx)
            # vely = math.sin(bullet[0])*10
            bullet[1] += 10
            # print(bullet)
            # bullet[2] += 0
            if bullet[1]<-64 or bullet[1]>640 :
                arrows.pop(index)
            index += 1
            for projectile in arrows:
                # print(projectile)
                # arrow1 = pygame.transform.rotate(arrow, 0)
                screen.blit(arrow, (projectile[1], projectile[2]))

        # 6.3 - Draw badgers
        if badtimer == 0:
            badguys.append([640, random.randint(50, 430)])
            badtimer = 100 - (badtimer1 * 2)
            if badtimer1 >= 35:
                badtimer1 = 35
            else:
                badtimer1 += 5
        index = 0
        for badguy in badguys:
            if badguy[0] < -64:
                badguys.pop(index)
            badguy[0] -= 3

            # 6.3.1 - Attack castle
            badrect = pygame.Rect(badguyimg.get_rect())
            badrect.top = badguy[1]
            badrect.left = badguy[0]
            if badrect.left < 64:
                hit.play()
                healthvalue -= random.randint(5,20)
                score -= 10 #########
                badguys.pop(index)

            # 6.3.2 - Check for collisions
            index1 = 0
            for bullet in arrows:
                bullrect = pygame.Rect(arrow.get_rect())
                bullrect.left = bullet[1]
                bullrect.top = bullet[2]
                if badrect.colliderect(bullrect):
                    enemy.play()
                    acc[0] += 1
                    score += 5
                    badguys.pop(index)
                    arrows.pop(index1)
                index1 += 1
            # 6.3.3 - Next bad guy
            index += 1
        for badguy in badguys:
            screen.blit(badguyimg, badguy)
        # 6.4 - Draw score
        font = pygame.font.Font(None, 24)
        scoretext = font.render(str(score), True, (0,0,0))
        textRect = scoretext.get_rect()
        textRect.topright = [635, 5]
        screen.blit(scoretext, textRect)

        # 6.5 - Draw health bar
        screen.blit(healthbar, (5, 5))
        for health1 in range(healthvalue):
            screen.blit(health, (health1 + 8, 8))
        # 7 - update the screen
        pygame.display.flip()
        # 8 - loop through the events
        for event in pygame.event.get():
            # check if the event is the X button
            if event.type == KEYDOWN:
                if event.key == K_w:
                    keys[0] = True
                elif event.key == K_s:
                    keys[1] = True

                elif event.key == K_SPACE:
                    shoot.play()
                    score -= 1
                    position = pygame.mouse.get_pos()
                    acc[1] += 1
                    arrows.append([math.atan2(position[1] - (playerpos1[1] + 32), position[0] - (playerpos1[0] + 26)),
                                   playerpos1[0] + 32, playerpos1[1] + 32])

            if event.type == KEYUP:
                if event.key == K_w:
                    keys[0] = False
                elif event.key == K_s:
                    keys[1] = False

            if event.type == QUIT:
                # if it is quit the game
                pygame.quit()
                exit(0)

        # 9 - Move player
        if keys[0]:
            playerpos[1] -= 5
        elif keys[1]:
            playerpos[1] += 5

        a = np.random.rand(1)


        if a[0] <= 0.45:
            playerpos[1] -= 5
        elif a[0] <= 0.9:
            playerpos[1] += 5
        else :
            shoot.play()
            score -= 1
            position = pygame.mouse.get_pos()
            acc[1] += 1
            arrows.append([math.atan2(position[1] - (playerpos1[1] + 32), position[0] - (playerpos1[0] + 26)),
                           playerpos1[0] + 32, playerpos1[1] + 32])

        badtimer -= 1

        # 10 - Win/Lose check
        if pygame.time.get_ticks() >= 90000:
            running = 0
            exitcode = 1
        if healthvalue <= 0:
            running = 0
            exitcode = 0
        if acc[1] != 0:
            accuracy = acc[0]*1.0/acc[1]*100
        else:
            accuracy = 0
        # a = pygame.surfarray.array_alpha(screen)
        # a = pygame.surfarray.array_colorkey(screen)
        # print(a[a==255])

        a = pygame.surfarray.array3d(screen)


        aa = rebin(rgb2gray(a), (64,48))

        aaa = rgb2gray(a)

        # surf = pygame.surfarray.make_surface(aaa)
        # display = pygame.display.set_mode((640, 480))
        # screen.blit(surf, (0, 0))
        # pygame.display.update()
        # pygame.surfarray.blit_array(screen, aaa)
        # state.append(aa)
        #
        # if len(state) == 2 :
        #     aaa = state[1] - state[0]
        #     print(aaa)

        #######################
        # font = pygame.font.Font(None, 24)
        # survivedtext = font.render(str((90000-pygame.time.get_ticks())//60000)+":"+str((90000-pygame.time.get_ticks())//1000%60).zfill(2), True, (0,0,0))
        # textRect = survivedtext.get_rect()
        # textRect.topright = [635, 5]
        # screen.blit(survivedtext, textRect)
        # accuracy = acc[0] * 1.0 / (acc[1] * 100 + 1e-7)
        # pygame.font.init()

    # 11 - Win/lose display
    if exitcode == 0:
        pygame.font.init()
        font = pygame.font.Font(None, 24)
        text = font.render("Accuracy: "+str(accuracy)+"%", True, (255, 0, 0))
        textRect = text.get_rect()
        textRect.centerx = screen.get_rect().centerx
        textRect.centery = screen.get_rect().centery+24
        screen.blit(gameover, (0, 0))
        screen.blit(text, textRect)
    else:
        pygame.font.init()
        font = pygame.font.Font(None, 24)
        text = font.render("Accuracy: "+str(accuracy)+"%", True, (0, 255, 0))
        textRect = text.get_rect()
        textRect.centerx = screen.get_rect().centerx
        textRect.centery = screen.get_rect().centery+24
        screen.blit(youwin, (0, 0))
        screen.blit(text, textRect)

    # while 1:
    #
    #     for event in pygame.event.get():
    #         if event.type == QUIT:
    #             pygame.quit()
    #             exit(0)
    #     pygame.display.flip()