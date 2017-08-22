import gym
import scipy
from gym import wrappers


#def rgb2gray(rgb):
#    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def image_cut(img, left_border = 5, right_border=  155, top_border = 65, buttom_border = 190):
    return img[top_border:buttom_border,left_border:right_border,-1]

def img_preprocess(img):
    img=image_cut(img)
    img=scipy.misc.imresize(img, (84,84), interp='nearest')
    return img

class skiing():
    def __init__(self):
        self.env = gym.make('Skiing-v0')
        self.env = wrappers.Monitor(self.env, '/tmp/skiing-experiment-0', force=True)
    def reset(self):
        observation = self.env.reset()
        return img_preprocess(observation)

    def step(self,action):
        # a = 0 - go down
        # a = 1 - go right
        # a = 2 - go left
        observation, reward, done, info = self.env.step(action)
        return img_preprocess(observation), reward, done, info