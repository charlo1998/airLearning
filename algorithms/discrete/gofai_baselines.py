
from random import choice
import sys
import gym
import math
import numpy as np

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
            action = agent.predict(obs,env)
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
        if (self.action_space != settings.action_discretization*4):
            print(f"wrong action space! should be 16*4 but is {self.action_space}")
        if (self.action_space != settings.action_discretization+2):
            print(f"wrong observation space! should be 16+2 but is {self.observation_space}")
        self.angle = 360/settings.action_discretization
        self.heading_coeff = 1
        self.safety_coeff = 2
        self.safety_dist = 4.0
        self.previous = 0




    def predict(self, obs, env):
        '''
        observation is in the form [angle, d_goal, y_vel, x_vel, d1, d2, ..., d16] where d1 starts at 180 deg and goes ccw, velocities are in drone's body frame ref
        actions are distributed as following:
        0-15: small circle
        16-31: medium small circle
        32-47: medium big circle
        48-63: big circle
        '''

        obs = obs[0][0] #flattening the list
        obs[4:] = 100**obs[4:] #reconverting from normalized to real values
        obs[1] = 100**obs[1]

        goal_angle = obs[0]*math.pi
        goal_distance = obs[1]
        x_vel = obs[3]
        y_vel = obs[2]
        sensors = obs[4:]
        
        angles =  np.arange(-180,180,self.angle)*math.pi/180
        sensors = np.concatenate((sensors,sensors)) #this way we can more easily slice the angles we want
        angles = np.concatenate((angles,angles))

        bestBenefit = -1000
        action = 0
        
        #print(f"angle to goal: {goal_angle*180/math.pi}")
        #print(f"distance to goal: {goal_distance}")
        #print(f"sensors: {np.round(sensors,1)}")

        for i in range(settings.action_discretization*4): #settings.action_discretization*4
            idx = 15 + 12 - i%settings.action_discretization #in the action space, the circle starts at 90 deg and goes cw
            objects = sensors[idx-3:idx+5] #only consider the obstacles in the direction we're going
            thetas = angles[idx-3:idx+5]


            #computing new distance to goal
            travel_dist = settings.base_speed*2**(i//settings.action_discretization)*(settings.mv_fw_dur) #travelled distance can be 0.5, 1, 2, or 4 times duration
            x_dest = travel_dist*math.cos(thetas[4]) + x_vel * 0.05 # correcting for current speed since change in speed isn't instantaneous
            y_dest = travel_dist*math.sin(thetas[4]) + y_vel * 0.05
            x_goal = goal_distance*math.sin(goal_angle) #reference frame for angle to goal is inverted
            y_goal = goal_distance*math.cos(goal_angle)
            new_dist = np.sqrt((x_goal-x_dest)**2+(y_goal-y_dest)**2)

            #computing the closest obstacle to the trajectory
            minDist = self.safety_dist #change to self.safety_dist
            for object,angle in zip(objects,thetas):
                x_obj = object*math.cos(angle+self.angle/2)
                y_obj = object*math.sin(angle+self.angle/2)
                dist = self.shortest_distance_on_trajectory(x_obj,y_obj,x_dest,y_dest)
                if dist < minDist:
                    minDist = dist

            #computing the benefit
            benefit = self.heading_coeff*(goal_distance-new_dist) - self.safety_coeff*(self.safety_dist - minDist)**2
            if benefit > bestBenefit:
                bestBenefit = benefit
                action =i



        ### -----------printing info on the chosen action-------------------------------------------------------------
        idx = 15 + 12 - action%settings.action_discretization #in the action space, the circle starts at 90 deg and goes cw
        objects = sensors[idx-3:idx+5] #only consider the obstacles in the direction we're going
        thetas = angles[idx-3:idx+5]

        #computing new distance to goal
        travel_dist = settings.base_speed*2**(i//settings.action_discretization)*(settings.mv_fw_dur) #travelled distance can be 0.5, 1, 2, or 4 times duration
        x_dest = travel_dist*math.cos(thetas[4]) + x_vel * 0.01 # correcting for current speed since change in speed isn't instantaneous
        y_dest = travel_dist*math.sin(thetas[4]) + y_vel * 0.01
        x_goal = goal_distance*math.sin(goal_angle) #reference frame for angle to goal is inverted
        y_goal = goal_distance*math.cos(goal_angle)
        new_dist = np.sqrt((x_goal-x_dest)**2+(y_goal-y_dest)**2)

        #computing the closest obstacle to the trajectory
        minDist = self.safety_dist #change to self.safety_dist
        for object,angle in zip(objects,thetas):
            x_obj = object*math.cos(angle+self.angle/2)
            y_obj = object*math.sin(angle+self.angle/2)
            dist = self.shortest_distance_on_trajectory(x_obj,y_obj,x_dest,y_dest)
            if dist < minDist:
                minDist = dist

        #computing the benefit
        benefit = self.heading_coeff*(goal_distance-new_dist) - self.safety_coeff*(self.safety_dist - minDist)**2

        print(f"min distance in chosen trajectory: {minDist}")
        #print(f"objects: {np.round(objects,1)}")
        #print(f"angles: {thetas*180/math.pi}")
        #print(f"observed goal (relative): {[y_goal,x_goal]}")
        #print(f"destination: {[y_dest, x_dest]}")
        now = env.airgym.drone_pos()
        now[0] += y_dest
        now[1] += x_dest
        #print(f"destination: {np.round(now,2)}")
        #now[0] += y_vel * 0.3
        #now[1] += x_vel * 0.3
        print(f"corrected destination: {np.round(now,2)}")
        #print(f"new_dist: {new_dist}")
        #---------------------------------------------


        #action = 1-self.previous
        #self.previous = action

        
        return action

    def shortest_distance_on_trajectory(self, x1,y1,x2,y2):
        """
        finds de the shortest distance to (x1,y1) by moving along the (x2,y2) line segment (from the origin)
        """

        dot = x1*x2 + y1*y2
        norm = x2*x2 + y2*y2
        if norm == 0:
            norm = 0.0001

        param = -1
        param = dot/norm

        if param < 0:
            xx = 0
            yy = 0
        elif param > 1:
            xx = x2
            yy = y2
        else:
            xx = param*x2
            yy = param*y2

        dx = x1 - xx
        dy = y1 - yy

        return math.sqrt(dx**2+dy**2)

    def state_machine(self, obs):
                #if self.avoiding: #avoid mode
        #    print(f"counter: {self.avoidance_counter}")
        #    if obs[0] <= 0.1: #still an obstacle in front
        #        action = 10
        #        self.avoidance_counter = self.avoidance_length
        #        print("turn right fast")
        #    elif self.avoidance_counter > self.avoidance_length/2: #continue turning right to avoid obstacle
        #        action = 11
        #        print("turn right")
        #        self.avoidance_counter -= 1
        #    elif self.avoidance_counter > 0: #go forward to avoid obstacle
        #        action = 3
        #        self.avoidance_counter -= 1
        #        print("straight")
        #    else:
        #        self.avoiding = False
        #        action = 2
        #        print("straight")
        #        self.avoidance_counter = self.avoidance_length
        #else:
        #    if (obs[0] <= 0.25): #first check if obstacles are in front, and enter avoid mode.
        #        self.avoiding = True
        #        self.avoidance_counter = self.avoidance_length
        #        print("entering avoidance mode!")
        #        action = 8
        #    elif (obs[4] <= -0.05):  #if no obstacles, try to align to the goal (turn left)
        #        action = 17
        #        print("turn left")
        #    elif (obs[4] >= 0.05):  #if no obstacles, try to align to the goal (turn right)
        #        action = 12
        #        print("turn right")
        #    elif (obs[5] >= 0.5): #if aligned and far, go straight fast
        #        action = 1
        #        print("straight fast")
        #    else: #if aligned and near, go straight slow
        #        action = 4
        #        print("straight slow")

        return 0

       
