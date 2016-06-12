#!/usr/bin/python
#####################################################################
#
#  This is the superclass for the Deep Learning Doom Player. 
#  
#####################################################################


from __future__ import print_function

from Tkconstants import OFF
from collections import deque
import cv2
import os
from random import choice
import random
import sys
from time import sleep
from time import time
from tqdm import *

import numpy as np
import tensorflow as tf
from vizdoom import Button
from vizdoom import DoomGame
from vizdoom import GameVariable
from vizdoom import Mode
from vizdoom import ScreenFormat
from vizdoom import ScreenResolution


class DeepDoom:
    """
    This is the superclas where all hyperparameter, the deep network, 
    the action and learning methodes are defined.
    """
    
    num_of_possible_actions = 3  # number of valid actions. In this case up, still and down
    future_reward_discount = 0.99  # decay rate of past observations
    observation_steps = 2500  # time steps to observe before training
    explore_steps = 1000000  # frames over which to anneal epsilon
    initial_random_action_prob = 1.0  # starting chance of an action being random
    final_random_action_prob = 0.01  # final chance of an action being random
    memory_size = 500000  # number of observations to remember
    mini_batch_size = 64  # size of mini batches
    state_frames = 1  # number of frames to store in the state
    resized_screen_x, resized_screen_y = (40, 40)
    obs_last_state_index, obs_action_index, obs_reward_index, obs_current_state_index, obs_terminal_index = range(5)
    learn_rate = 1e-5 
    checkpoint_path="./checkpoint/" # this is the path where the checkpoints will be saved
    episodes = 50
    training_steps_per_epoch = 100000
    
    
    def convert_image(self, img):
        """
        This function converts the image  and resizes it.
        To help debuging one can use the imsho function von cv2: cv2.imshow('show image',img) to show the image 
        and to wait you can use cv2.waitKey(0)
        
        """
        img = img[0].astype(np.float32) / 255.0
        img = cv2.resize(img, (self.resized_screen_x, self.resized_screen_y))
        
        return img
        
    
    
    
    
    
    def __init__ (self):
        """
        do i need this hier????     
        
        """
        
        # set scroe on start to 0
        self.last_score = 0
        
        #get tensorflow session, set up network
        self.session = tf.Session()
        self.input_layer, self.output_layer = self.create_network()
        
        self.action = tf.placeholder("float", [None, self.num_of_possible_actions])
        self.target = tf.placeholder("float", [None])

        readout_action = tf.reduce_sum(tf.mul(self.output_layer, self.action), reduction_indices=1)

        cost = tf.reduce_mean(tf.square(self.target - readout_action))
        self.train_operation = tf.train.AdamOptimizer(self.learn_rate).minimize(cost)


        # deque https://pymotw.com/2/collections/deque.html
        self.observations = deque()
        self.last_scores = deque()
        
        
        # set the first action to do nothing
        self.last_action = np.zeros(self.num_of_possible_actions)
        self.last_action[1] = 1

        self.last_state = None
        self.probability_of_random_action = self.initial_random_action_prob
        self.time = 0

        #start the deep learning network 
        self.session.run(tf.initialize_all_variables())
        self.saver = tf.train.Saver()
        
    
    def start(self):
        """
         this will get passed hier
        """
        
        
        # Create DoomGame instance. It will run the game and communicate with you.
        print ("Initializing doom...")
        game = DoomGame()
        game.load_config("./examples/config/learningtensorflow.cfg")
        game.init()
        print ("Doom initialized.")
        
        
        for epoch in range(DeepDoom.episodes):
            print ("\nEpoch", epoch)
            train_time = 0
            train_episodes_finished = 0
            train_loss = []
            train_rewards = []
            
                        
                
            #if epoch > 3:
            #     self.saver.save(self.session, self.checkpoint_path, global_step=epoch )

            train_start = time()

            game.new_episode()
        
            for learning_step in tqdm(range(DeepDoom.training_steps_per_epoch)):
        



                if game.is_episode_finished():
                    #print("game is finished")
                    r = game.get_total_reward()
                    train_rewards.append(r)
                    game.new_episode()
                    train_episodes_finished += 1
                    self.last_state = None
                    #sleep(sleep_time)

                
                # first frame must be handled differently
                if self.last_state is None:
                    #print ("ich bin hier")
                    # the last_state will contain the image data from the last self.state_frames frames
                    self.last_state = np.stack(tuple(self.convert_image(game.get_state().image_buffer) for _ in range(self.state_frames)), axis=2)
                    continue
                   
                
                
                
                reward = game.make_action(DeepDoom.define_keys_to_action_pressed(self.last_action), 7)
           
                
                reward *= 0.01
                
               
                
                
                
                
                #if screen_array is not None:   
                imagebuffer = game.get_state().image_buffer
                
                
                #if reward > 0 and imagebufferlast is not None:
                #    img = imagebufferlast[0].astype(np.float32) / 255.0
                #    img = cv2.resize(img, (80, 80))
                #    cv2.imshow('asd', img)        
                #    cv2.waitKey(0)
                
                if imagebuffer is None:
                    terminal = True
                    #print(reward)
                    screen_resized_binary =  np.zeros((40,40))
                    
                imagebufferlast = imagebuffer 
                    
                if imagebuffer is not None: 
                    terminal = False
                    screen_resized_binary = self.convert_image(imagebuffer)
                
                # add dimension
                screen_resized_binary = np.expand_dims(screen_resized_binary, axis=2)
                
                #print(screen_resized_binary.shape)
                #print(self.last_state[:, :, 1:].shape)
                
                
                current_state = np.append(self.last_state[:, :, 1:], screen_resized_binary, axis=2)
                
                
                self.observations.append((self.last_state, self.last_action, reward, current_state, terminal))


                #zeugs.write("oberservations %s \n" %len(self.observations))

                if len(self.observations) > self.memory_size:
                    self.observations.popleft()
                    #sleep(sleep_time)

                # only train if done observing
                if len(self.observations) > self.observation_steps:
                    #print("train")
                    self.train()
                    self.time += 1
                
                self.last_state = current_state

                self.last_action = self.choose_next_action()
                
                
                if self.probability_of_random_action > self.final_random_action_prob \
                        and len(self.observations) > self.observation_steps:
                    self.probability_of_random_action -= \
                        (self.initial_random_action_prob - self.final_random_action_prob) / self.explore_steps
                        
                
                
                
            print (train_episodes_finished, "training episodes played.")
            print ("Training results:")

            train_rewards = np.array(train_rewards)
            
            train_end = time()
            train_time = train_end - train_start
            mean_loss = np.mean(train_loss)


            print ("mean:", train_rewards.mean(), "std:", train_rewards.std(), "max:", train_rewards.max(), "min:", train_rewards.min(),  "epsilon:", self.probability_of_random_action)
            print ("t:", str(round(train_time, 2)) + "s")
            
            
        
        # It will be done automatically anyway but sometimes you need to do it in the middle of the program...
        game.close()
        self.last_state = None
        
        

    

    @staticmethod
    def create_network():
        """
        The definition of the deep neural convoluting network
        """
        
        convolution_weights_1 = tf.Variable(tf.truncated_normal([8, 8, DeepDoom.state_frames, 32], stddev=0.01))
        convolution_bias_1 = tf.Variable(tf.constant(0.01, shape=[32]))

        convolution_weights_2 = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev=0.01))
        convolution_bias_2 = tf.Variable(tf.constant(0.01, shape=[64]))

        feed_forward_weights_1 = tf.Variable(tf.truncated_normal([256, 256], stddev=0.01))
        feed_forward_bias_1 = tf.Variable(tf.constant(0.01, shape=[256]))

        feed_forward_weights_2 = tf.Variable(tf.truncated_normal([256, DeepDoom.num_of_possible_actions], stddev=0.01))
        feed_forward_bias_2 = tf.Variable(tf.constant(0.01, shape=[DeepDoom.num_of_possible_actions]))

        input_layer = tf.placeholder("float", [None, DeepDoom.resized_screen_x, DeepDoom.resized_screen_y,
                                               DeepDoom.state_frames])

        hidden_convolutional_layer_1 = tf.nn.relu(
            tf.nn.conv2d(input_layer, convolution_weights_1, strides=[1, 4, 4, 1], padding="SAME") + convolution_bias_1)

        hidden_max_pooling_layer_1 = tf.nn.max_pool(hidden_convolutional_layer_1, ksize=[1, 2, 2, 1],
                                                    strides=[1, 2, 2, 1], padding="SAME")

        hidden_convolutional_layer_2 = tf.nn.relu(
            tf.nn.conv2d(hidden_max_pooling_layer_1, convolution_weights_2, strides=[1, 2, 2, 1],
                         padding="SAME") + convolution_bias_2)

        hidden_max_pooling_layer_2 = tf.nn.max_pool(hidden_convolutional_layer_2, ksize=[1, 2, 2, 1],
                                                    strides=[1, 2, 2, 1], padding="SAME")

        hidden_convolutional_layer_3_flat = tf.reshape(hidden_max_pooling_layer_2, [-1, 256])

        final_hidden_activations = tf.nn.relu(
            tf.matmul(hidden_convolutional_layer_3_flat, feed_forward_weights_1) + feed_forward_bias_1)

        output_layer = tf.matmul(final_hidden_activations, feed_forward_weights_2) + feed_forward_bias_2

        return input_layer, output_layer
    
    
    def choose_next_action(self):
        """
        Choosing the next action random when exploring or on Q values.
        This is used by the learning agent.
        """
        new_action = np.zeros([self.num_of_possible_actions])

        if (random.random() <= self.probability_of_random_action):
            # choose an action randomly
            action_index = random.randrange(self.num_of_possible_actions)
        else:
            # choose an action given our last state
            readout_t = self.session.run(self.output_layer, feed_dict={self.input_layer: [self.last_state]})[0]

            # chose the action with the highest Q Value
            action_index = np.argmax(readout_t)

        new_action[action_index] = 1
        return new_action
    
    
    
    def choose_next_action_only_on_q(self):
        """
        Choosing the next action only on Q values.
        This is used by the playing agent.
        """
        new_action = np.zeros([self.num_of_possible_actions])
       
        # choose an action given our last state
        readout_t = self.session.run(self.output_layer, feed_dict={self.input_layer: [self.last_state]})[0]

        # chose the action with the highest Q Value
        action_index = np.argmax(readout_t)

        new_action[action_index] = 1
        return new_action
    
    
    def train(self):
        """
        Method for sampling a minibatch and training the conv network
        """
        
        # sample a mini_batch to train on
        mini_batch = random.sample(self.observations, self.mini_batch_size)
        previous_states = [d[self.obs_last_state_index] for d in mini_batch]
        actions = [d[self.obs_action_index] for d in mini_batch]
        rewards = [d[self.obs_reward_index] for d in mini_batch]
        current_states = [d[self.obs_current_state_index] for d in mini_batch]
        agents_expected_reward = []
        # do a feed forward pass to get the action q values fpr the new state
        agents_reward_per_action = self.session.run(self.output_layer, feed_dict={self.input_layer: current_states})
        for i in range(len(mini_batch)):
            if mini_batch[i][self.obs_terminal_index]:
                # this was a terminal frame so there is no future reward...
                agents_expected_reward.append(rewards[i])
            else:
                agents_expected_reward.append(
                    rewards[i] + self.future_reward_discount * np.max(agents_reward_per_action[i]))

        # learn that these actions in these states lead to this reward

        self.session.run(self.train_operation, feed_dict={
            self.input_layer: previous_states,
            self.action: actions,
            self.target: agents_expected_reward})

        
        
    @staticmethod
    # Define some actions. Each list entry corresponds to declared buttons:
    # MOVE_LEFT, MOVE_RIGHT, ATTACK
    # 5 more combinations are naturally possible but only 3 are included     
    # actions = [[True,False,False],[False,True,False],[False,False,True]]
    def define_keys_to_action_pressed(action_set):
        if action_set[0] == 1:
            return [True,False,False] # Move_Left
        elif action_set[1] == 1:
            return [False,True,False] # Move_right
        elif action_set[2] == 1:
            return [False,False,True] # Attack
        raise Exception("Unexpected action")    

if __name__ == '__main__':
    # to see a trained network add the args checkpoint_path="deep_q_half_pong_networks_40x40_8" and
    # playback_mode="True"
    player = DeepDoom()
    player.start()









