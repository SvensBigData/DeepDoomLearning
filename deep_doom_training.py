#!/usr/bin/python
#####################################################################
#
#  This is the deep doom traing class
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
from deep_doom import DeepDoom


class DeepDoomTraining(DeepDoom):
    """
    This is the clas for deep doom training
    """
  
    def __init__ (self):
        """
        do i need this hier????     
        
        """
        
        # set scroe on train to 0
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

        #train the deep learning network 
        self.session.run(tf.initialize_all_variables())
        self.saver = tf.train.Saver()
        
    
    def train(self):
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
        
        

    


if __name__ == '__main__':
    # to see a trained network add the args checkpoint_path="deep_q_half_pong_networks_40x40_8" and
    # playback_mode="True"
    player = DeepDoomTraining()
    player.train()









