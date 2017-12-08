from PIL import Image, ImageDraw
from numpy import genfromtxt


g = open('data\\test_data.csv','r')
temp = genfromtxt(g, delimiter = ',', dtype=int)
im = Image.fromarray(temp).convert('RGB')
pix = im.load()
rows, cols = im.size
for x in range(cols):
    for y in range(rows):
        print(str(x) + " " + str(y))
        pix[x,y] = (int(temp[y,x] // 256 // 256 % 256),int(temp[y,x] // 256  % 256),int(temp[y,x] % 256))
im.save(g.name[0:-4] + '.jpeg')