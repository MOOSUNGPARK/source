import gym
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.models import Model
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Flatten, Input
from keras.layers.convolutional import Conv2D
from keras import backend as K

global episode
episode = 0
EPISODES = 99999999
env_name = "SpaceInvaders-v0"
K.set_learning_phase(1)



class Agent:
    def __init__(self, action_size):
        self.state_size = (84, 84, 4)
        self.action_size = action_size

        self.discount_factor = 0.99
        self.no_op_steps = 30

        self.actor, self.critic = self.build_model()

    def build_model(self):
        input = Input(shape=self.state_size)
        conv1 = Conv2D(16, (8, 8), strides=(4, 4))(input)
        batch1 = BatchNormalization(axis=1)(conv1)
        act1 = PReLU()(batch1)
        conv2 = Conv2D(32, (4, 4), strides=(2, 2))(act1)
        batch2 = BatchNormalization(axis=1)(conv2)
        act2 = PReLU()(batch2)
        conv = Flatten()(act2)
        fc = Dense(256, activation='relu')(conv)

        policy = Dense(self.action_size, activation='softmax')(fc)
        value = Dense(1, activation='linear')(fc)

        actor = Model(inputs=input, outputs=policy)
        critic = Model(inputs=input, outputs=value)

        actor.summary()
        critic.summary()

        return actor, critic

    def get_action(self, history):
        history = np.float32(history / 255.)
        policy = self.actor.predict(history)[0]

        action_index = np.argmax(policy)
        return action_index

    def load_model(self, name):
        self.actor.load_weights(name)

def pre_processing(next_observe, observe):
    processed_observe = np.maximum(next_observe, observe)
    processed_observe = np.uint8(
        resize(rgb2gray(processed_observe), (84, 84), mode='constant') * 255)
    return processed_observe


if __name__ == "__main__":
    env = gym.make(env_name)
    agent = Agent(action_size=3)
    agent.load_model("save_model3/invader4_actor.h5")

    step = 0
    while episode < EPISODES:
        done = False
        dead = False

        score, start_life = 0, 3
        observe = env.reset()
        next_observe = observe

        state = pre_processing(next_observe, observe)
        history = np.stack((state, state, state, state), axis=2)
        history = np.reshape([history], (1, 84, 84, 4))

        while not done:
            env.render()
            step += 1
            observe = next_observe

            action = agent.get_action(history)


            if action == 1:
                fake_action = 2
            elif action == 2:
                fake_action = 3
            else:
                fake_action = 1

            if dead:
                fake_action = 1
                dead = False

            next_observe, reward, done, info = env.step(fake_action)

            next_state = pre_processing(next_observe, observe)
            next_state = np.reshape([next_state], (1, 84, 84, 1))
            next_history = np.append(next_state, history[:, :, :, :3], axis=3)

            if start_life > info['ale.lives']:
                dead = True
                reward = -1
                start_life = info['ale.lives']

            score += reward

            # if agent is dead, then reset the history
            if dead:
                history = np.stack(
                    (next_state, next_state, next_state, next_state), axis=2)
                history = np.reshape([history], (1, 84, 84, 4))
            else:
                history = next_history

            # if done, plot the score over episodes
            if done:
                episode += 1
                print("episode:", episode, "  score:", score, "  step:", step)
                step = 0