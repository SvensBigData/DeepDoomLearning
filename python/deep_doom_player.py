#!/usr/bin/python
#####################################################################
#
#  This is the deep doom playing class
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

from deep_doom import DeepDoom


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



class DeepDoomPlayer(DeepDoom):

    def __init__ (self):
        
        # set scroe on play to 0
        self.last_score = 0
        
        #get tensorflow session, set up network
        self.session = tf.Session()
        self.input_layer, self.output_layer = self.create_network()
        
        self.action = tf.placeholder("float", [None, self.num_of_possible_actions])
        self.target = tf.placeholder("float", [None])

        # deque https://pymotw.com/2/collections/deque.html
        self.observations = deque()
        self.last_scores = deque()
       
         # set the first action to do nothing
        self.last_action = np.zeros(self.num_of_possible_actions)
        self.last_action[1] = 1

        self.last_state = None
       
        self.time = 0

        #play the deep learning network 
        self.session.run(tf.initialize_all_variables())
        self.saver = tf.train.Saver()
        
               
        checkpoint = tf.train.get_checkpoint_state(DeepDoom.checkpoint_path)
       
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            print("Loaded checkpoints %s" % checkpoint.model_checkpoint_path)
        else:
            print("Could not load any checkpoint exit now")
            sys.exit(0)
        
        
        
    def play(self):
        
        # Create DoomGame instance. It will run the game and communicate with you.
        print ("Initializing doom...")
        game = DoomGame()

        game.load_config("./examples/config/deepdoomplayer.cfg")
        game.init()
        print ("Doom initialized.")
 
        episodes = 1
        training_steps_per_epoch = 100

        sleep_time = 0.100

        train_episodes_finished = 0
        train_rewards = []
        
        for epoch in range(episodes):
           
            train_loss = []
            
            game.new_episode()
        
            while(train_episodes_finished < 20 ):
        
                sleep(sleep_time)   

                if game.is_episode_finished():
                    
                    r = game.get_total_reward()
                    train_rewards.append(r)
                    game.new_episode()
                    train_episodes_finished += 1
                    self._last_state = None
                    self.last_action[1] = 1

                # first frame must be handled differently
                if self.last_state is None:
                    # the _last_state will contain the image data from the last self.state_frames frames
                    self.last_state = np.stack(tuple(self.convert_image(game.get_state().image_buffer) for _ in range(self.state_frames)), axis=2)
                    continue

                
                reward = game.make_action(DeepDoomPlayer.define_keys_to_action_pressed(self.last_action), 7)
           
                reward *= 0.01

                imagebuffer = game.get_state().image_buffer

                if imagebuffer is None:
                    terminal = True
                    screen_resized_binary =  np.zeros((40,40))
                    
                imagebufferlast = imagebuffer 
                    
                if imagebuffer is not None: 
                    terminal = False
                    screen_resized_binary = self.convert_image(imagebuffer)
                
                # add dimension
                screen_resized_binary = np.expand_dims(screen_resized_binary, axis=2)

                current_state = np.append(self.last_state[:, :, 1:], screen_resized_binary, axis=2)

                self.last_state = current_state

                self.last_action = self.choose_next_action_only_on_q()

            print (train_episodes_finished, "training episodes played.")
            print ("Training results:")
            
            train_rewards = np.array(train_rewards)
  
            print ("mean:", train_rewards.mean(), 
                   "std:", train_rewards.std(), 
                   "max:", train_rewards.max(), 
                   "min:", train_rewards.min())
           
            
        # It will be done automatically anyway but sometimes you need to do it in the middle of the program...
        game.close()
        self._last_state = None


if __name__ == '__main__':
    # to see a trained network add the args checkpoint_path="deep_q_half_pong_networks_40x40_8" and
    # playback_mode="True"
    
    player = DeepDoomPlayer()
    player.play()









