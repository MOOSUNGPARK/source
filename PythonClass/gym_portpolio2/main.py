import gym
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import tensorflow as tf
from PythonClass.gym_portpolio2.experience_buffer import experience_buffer
from PythonClass.gym_portpolio2.dqn import Qnetwork
from PythonClass.gym_portpolio2.skiing import skiing
import os




def processState(states):
    return np.reshape(states,[7056])

def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars/2]):
        op_holder.append(tfVars[idx+total_vars/2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars/2].value())))
    return op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)

game = skiing()
batch_size = 32 #How many experiences to use for each training step.
update_freq = 4 #How often to perform a training step.
y = .99 #Discount factor on the target Q-values
startE = 1 #Starting chance of random action
endE = 0.1 #Final chance of random action
anneling_steps = 10000. #How many steps of training to reduce startE to endE.
num_episodes = 10000 #How many episodes of game environment to train network with.
pre_train_steps = 10000 #How many steps of random actions before training begins.
max_epLength = 5000 #The max allowed length of our episode.
load_model = True #Whether to load a saved model.
test_model = True #Exit after "done" flag is True
path = "./dqn" #The path to save our model to.
h_size = 512 #The size of the final convolutional layer before splitting it into Advantage and Value streams.
tau = 0.001 #Rate to update target network toward primary network

tf.reset_default_graph()
mainQN = Qnetwork(h_size)
targetQN = Qnetwork(h_size)

init = tf.initialize_all_variables()

saver = tf.train.Saver()

trainables = tf.trainable_variables()

targetOps = updateTargetGraph(trainables, tau)

myBuffer = experience_buffer()

# Set the rate of random action decrease.
e = startE
stepDrop = (startE - endE) / anneling_steps

# create lists to contain total rewards and steps per episode
jList = []
rList = []
total_steps = 0

# Make a path for our model to be saved in.
if not os.path.exists(path):
    os.makedirs(path)

with tf.Session() as sess:
    if load_model == True:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    sess.run(init)
    updateTarget(targetOps, sess)  # Set the target network to be equal to the primary network.
    for i in range(num_episodes):
        episodeBuffer = experience_buffer()
        # Reset environment and get first new observation
        s = game.reset()
        s = processState(s)
        d = False
        rAll = 0
        j = 0
        # The Q-Network
        while j < max_epLength:  # If the agent takes longer than 200 moves to reach either of the blocks, end the trial.
            j += 1
            # Choose an action by greedily (with e chance of random action) from the Q-network
            if np.random.rand(1) < e or total_steps < pre_train_steps:
                a = np.random.randint(0, 3)
            else:
                a = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: [s]})[0]
            s1, r, d, _ = game.step(a)
            game.env.render()

            s1 = processState(s1)
            total_steps += 1
            episodeBuffer.add(
                np.reshape(np.array([s, a, r, s1, d]), [1, 5]))  # Save the experience to our episode buffer.

            if total_steps > pre_train_steps:
                if e > endE:
                    e -= stepDrop

                if total_steps % (update_freq) == 0:
                    trainBatch = myBuffer.sample(batch_size)  # Get a random batch of experiences.
                    # Below we perform the Double-DQN update to the target Q-values
                    Q1 = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: np.vstack(trainBatch[:, 3])})
                    Q2 = sess.run(targetQN.Qout, feed_dict={targetQN.scalarInput: np.vstack(trainBatch[:, 3])})
                    end_multiplier = -(trainBatch[:, 4] - 1)
                    doubleQ = Q2[range(batch_size), Q1]
                    targetQ = trainBatch[:, 2] + (y * doubleQ * end_multiplier)
                    # Update the network with our target values.
                    _ = sess.run(mainQN.updateModel,
                                 feed_dict={mainQN.scalarInput: np.vstack(trainBatch[:, 0]), mainQN.targetQ: targetQ,
                                            mainQN.actions: trainBatch[:, 1]})

                    updateTarget(targetOps, sess)  # Set the target network to be equal to the primary network.
            rAll += r
            s = s1

            if d == True:
                if test_model:
                    game.env.monitor.close()
                break


        # Get all experiences from this episode and discount their rewards.
        myBuffer.add(episodeBuffer.buffer)
        jList.append(j)
        rList.append(rAll)
        # Periodically save the model.
        if i % 1000 == 0:
            saver.save(sess, path + '/model-' + str(i) + '.cptk')
            print("Saved Model")
        if len(rList) % 10 == 0:
            print(total_steps, np.mean(rList[-10:]), e)
    saver.save(sess, path + '/model-' + str(i) + '.cptk')
print("Percent of succesful episodes: " + str(sum(rList) / num_episodes) + "%")
