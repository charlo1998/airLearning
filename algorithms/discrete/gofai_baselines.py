

import sys
import gym

import os
import tensorflow as tf
os.sys.path.insert(0, os.path.abspath('../../../settings_folder'))
import settings
import msgs
from gym_airsim.envs.airlearningclient import *
import callbacks
from stable_baselines.common.vec_env import DummyVecEnv
#from stable_baselines.common import make_vec_env #this yields an error
from keras.backend.tensorflow_backend import set_session

def setup(difficulty_level='default', env_name = "AirSimEnv-v42"):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.6
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    env = gym.make(env_name)
    env.init_again(eval("settings."+difficulty_level+"_range_dic"))

    # Vectorized environments allow to easily multiprocess training
    # we demonstrate its usefulness in the next examples
    vec_env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run
    # Parallel environments
    return env

def train(env, agent, checkpoint="C:/Users/charl/workspace/airlearning/airlearning-rl/data/A2C-B/model"):
    print("no need for training!")

def test(env):
    msgs.mode = 'test'

    agent = gofai(env)
    
    for i in range(settings.testing_nb_episodes_per_model):
        obs = env.reset()
        done = False
        while not done:
            action = agent.predict(obs)
            obs, rewards, done, info = env.step(action)

    #env loop rate logging
    if settings.profile:
        with open(os.path.join(settings.proj_root_path, "data", "env","env_log.txt"),
            "w") as f:
            f.write("loop_rate_list:" + str(env.loop_rate_list) + "\n")
            f.write("take_action_list:" + str(env.take_action_list) + "\n")
            f.write("clct_state_list:" + str(env.clct_state_list) + "\n")

        action_duration_file = os.path.join(settings.proj_root_path, "data", msgs.algo, "action_durations" + str(settings.i_run) + ".txt")
        with open(action_duration_file, "w") as f:
            f.write(str(env.take_action_list))



class gofai():
    '''
    naive implementation of a pursuing algorithm with obstacle avoidance.
    '''

    def __init__(self, env):
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        if (self.action_space != 25):
            print(f"wrong action space! should be 25 but is {self.action_space}")
        if (self.action_space != 6):
            print(f"wrong observation space! should be 6 but is {self.observation_space}")
        self.avoidance_length = 20
        self.avoidance_counter = 0
        self.avoiding = False


    def predict(self, obs):
        '''
        observation is in the form [angle, d_goal, d1, d2, d3, d4]
        actions are distributed as following:
        0-4: go straight with decreasing speed
        5-9: backup with decreasing speed
        10-14: yaw right with decreasing speed
        15-19: yaw left with decreasing speed
        '''

        obs = obs[0][0] #flattening the list

        if self.avoiding: #avoid mode
            print(f"counter: {self.avoidance_counter}")
            if obs[0] <= 0.1: #still an obstacle in front
                action = 10
                self.avoidance_counter = self.avoidance_length
                print("turn right fast")
            elif self.avoidance_counter > self.avoidance_length/2: #continue turning right to avoid obstacle
                action = 11
                print("turn right")
                self.avoidance_counter -= 1
            elif self.avoidance_counter > 0: #go forward to avoid obstacle
                action = 3
                self.avoidance_counter -= 1
                print("straight")
            else:
                self.avoiding = False
                action = 2
                print("straight")
                self.avoidance_counter = self.avoidance_length
        else:
            if (obs[0] <= 0.25): #first check if obstacles are in front, and enter avoid mode.
                self.avoiding = True
                self.avoidance_counter = self.avoidance_length
                print("entering avoidance mode!")
                action = 8
            elif (obs[4] <= -0.05):  #if no obstacles, try to align to the goal (turn left)
                action = 17
                print("turn left")
            elif (obs[4] >= 0.05):  #if no obstacles, try to align to the goal (turn right)
                action = 12
                print("turn right")
            elif (obs[5] >= 0.5): #if aligned and far, go straight fast
                action = 2
                print("straight fast")
            else: #if aligned and near, go straight slow
                action = 4
                print("straight slow")


        return action
       
