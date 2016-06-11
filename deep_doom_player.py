#!/usr/bin/python
#####################################################################
# This script presents how to use the most basic features of the environment.
# It configures the engine, and makes the agent perform random actions.
# It also gets current state and reward earned with the action.
# <episodes> number of episodes are played. 
# Random combination of buttons is chosen for every action.
# Game variables from state and last reward are printed.
# To see the scenario description go to "../../scenarios/README.md"
# 
#####################################################################
from __future__ import print_function
from vizdoom import DoomGame
from vizdoom import Mode
from vizdoom import Button
from vizdoom import GameVariable
from vizdoom import ScreenFormat
from vizdoom import ScreenResolution
# Or just use from vizdoom import *

from random import choice
from time import sleep
from time import time
import sys

from tqdm import *
import os
import random
from collections import deque

from basic_qlerntesorflow_one_frame import DeepQDoomPlayer


import tensorflow as tf
import numpy as np
import cv2
from Tkconstants import OFF

#garbage
terminal = False
downsampled_x = 80
downsampled_y = 80
sleep_time = 0.028
skiprate = 7



class DeepDoomPlayer(DeepQDoomPlayer):

    
      
    
    def __init__ (self):
        
        # set scroe on start to 0
        self.last_score = 0
        
        #get tensorflow session, set up network
        self._session = tf.Session()
        self._input_layer, self._output_layer = self._create_network()
        
        self._action = tf.placeholder("float", [None, self.ACTIONS_COUNT])
        self._target = tf.placeholder("float", [None])

        readout_action = tf.reduce_sum(tf.mul(self._output_layer, self._action), reduction_indices=1)

        cost = tf.reduce_mean(tf.square(self._target - readout_action))
        self._train_operation = tf.train.AdamOptimizer(self.LEARN_RATE).minimize(cost)


        # deque https://pymotw.com/2/collections/deque.html
        self._observations = deque()
        self._last_scores = deque()
        
        
         # set the first action to do nothing
        self._last_action = np.zeros(self.ACTIONS_COUNT)
        self._last_action[1] = 1

        self._last_state = None
        self._probability_of_random_action = self.INITIAL_RANDOM_ACTION_PROB
        self._time = 0

        #start the deep learning network 
        self._session.run(tf.initialize_all_variables())
        self._saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(self.checkpoint_path)
        
        
        self._saver.restore(self._session, checkpoint.model_checkpoint_path)
        print("Loaded checkpoints %s" % checkpoint.model_checkpoint_path)
        
        
        
    
    def start(self):
        
        # Create DoomGame instance. It will run the game and communicate with you.
        print ("Initializing doom...")
        game = DoomGame()

        game.load_config("./examples/config/deepdoomplayer.cfg")
        game.init()
        print ("Doom initialized.")
 
        episodes = 100000000
        training_steps_per_epoch = 100

        sleep_time = 0.100
                
        
        for epoch in range(episodes):
            print ("\nEpoch", epoch)
            
            train_episodes_finished = 0
            train_loss = []
            train_rewards = []


            train_start = time()

            game.new_episode()
        
            for learning_step in tqdm(range(training_steps_per_epoch)):
        

                sleep(sleep_time)   


                if game.is_episode_finished():
                    #print("game is finished")
                    r = game.get_total_reward()
                    train_rewards.append(r)
                    game.new_episode()
                    train_episodes_finished += 1
                    self._last_state = None
                    sleep(0.3)

                
                # first frame must be handled differently
                if self._last_state is None:
                    #print ("ich bin hier")
                    # the _last_state will contain the image data from the last self.STATE_FRAMES frames
                    self._last_state = np.stack(tuple(self.convert_image(game.get_state().image_buffer) for _ in range(self.STATE_FRAMES)), axis=2)
                    continue

                
                reward = game.make_action(DeepDoomPlayer._key_presses_from_action(self._last_action), 7)
           
                
                reward *= 0.01

                imagebuffer = game.get_state().image_buffer

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

                current_state = np.append(self._last_state[:, :, 1:], screen_resized_binary, axis=2)

                self._last_state = current_state

                self._last_action = self._choose_next_action_only_onq()

            print (train_episodes_finished, "training episodes played.")
            print ("Training results:")

            train_rewards = np.array(train_rewards)
          
            mean_loss = np.mean(train_loss)


            print ("mean:", train_rewards.mean(), "std:", train_rewards.std(), "max:", train_rewards.max(), "min:", train_rewards.min(),  "epsilon:", self._probability_of_random_action)
           
            
            
        
        # It will be done automatically anyway but sometimes you need to do it in the middle of the program...
        game.close()
        self._last_state = None


if __name__ == '__main__':
    # to see a trained network add the args checkpoint_path="deep_q_half_pong_networks_40x40_8" and
    # playback_mode="True"
    
    player = DeepDoomPlayer()
    player.start()









