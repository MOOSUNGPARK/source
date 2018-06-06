import re
import numpy as np
import cv2

def _number_key(s):
    return [ss for ss in re.split('([0-9]+)', s)]



print(_number_key('15kdkfls215'))


a = 'a/bc'

print(a.replace('a/','b'))


a = np.arange(5)
np.random.shuffle(a)
print(a)

img = cv2.imread('C:\\Users\\sunki\\Pictures\\pic1.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
img1 = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)[1]

img2 = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY_INV)[1]
img1 = img1.reshape([256, 256, 1])
img2 = img2.reshape([256, 256, 1])
img = np.concatenate((img1, img2), axis=2)
cv2.imwrite('C:\\Users\\sunki\\Pictures\\img1.jpg',img1)

cv2.imwrite('C:\\Users\\sunki\\Pictures\\img2.jpg',img2)
# cv2.imshow('img',img)
cv2.imwrite('C:\\Users\\sunki\\Pictures\\img.jpg',img)