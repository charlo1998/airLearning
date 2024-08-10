
from random import choice
import sys
import gym
import time

import os
import tensorflow as tf
os.sys.path.insert(0, os.path.abspath('../../../settings_folder'))
import settings
import msgs
from utils import gofai, APF
from tangent_bug import tangent_bug
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

def train(env, agent, checkpoint=os.path.expanduser("~") + "/workspace/airlearning/airlearning-rl/data/A2C-B/model"):
    print("no need for training!")

def test(env):
    msgs.mode = 'test'
    process_action_list = []
    cpu_times_list = []
    DWA = gofai()
    APF_planner = APF()
    bug = tangent_bug()
    

    for i in range(settings.testing_nb_episodes_per_model):
        print("resetting env")
        obs = env.reset()
        done = False
        while not done:
            begin = time.perf_counter()
            
            #print("--------------------------------------bug---------------------------------------------")
            goal = bug.predict(obs)
            bug_end = time.perf_counter()
            #print("--------------------------------------dwa---------------------------------------------")
            #action = DWA.predict(obs,goal)
            # -------------- skipping bug -----------
            print(f"bug goal: {goal}")
            goal = env.goal-env.airgym.drone_pos()
            dwa_goal = [goal[1], goal[0]] #inverted x and y
            print(f"relative goal: {dwa_goal}")
            # --------------------------------
            begin_CPU = time.process_time()
            #for i in range(100):
            #    action = APF_planner.predict(obs,goal)
            action =DWA.predict(obs,dwa_goal)
            end = time.perf_counter()
            end_CPU = time.process_time()

            #print(f"bug processing: {np.round((bug_end - begin)*1000)} ms")
            #print(f"dwa processing: {np.round((end - begin)*1000)} ms")
            print(f"APF processing: {np.round((end - bug_end)*1000,2)} ms")
            print(f"APF processing: {np.round((end_CPU - begin_CPU)*1000,4)} ms")
            
            #---------------------step by step mode----------------------
            #env.airgym.client.simPause(True)
            #answer = input()
            #env.airgym.client.simPause(False)
            obs, rewards, done, info = env.step(action)
            bug.done = done

            if settings.profile:
                process_action_list.append(end-bug_end)
                cpu_times_list.append(end_CPU-begin_CPU)

    #env loop rate logging
    if settings.profile:
        print(f"Average clock processing time: {sum(process_action_list)/len(process_action_list)*1000} ms")
        print(f"Average CPU processing time: {sum(cpu_times_list)/len(cpu_times_list)*1000} ms")
        with open(os.path.join(settings.proj_root_path, "data", "env","env_log.txt"),
            "w") as f:
            f.write("loop_rate_list:" + str(env.loop_rate_list) + "\n")
            f.write("take_action_list:" + str(env.take_action_list) + "\n")
            f.write("clct_state_list:" + str(env.clct_state_list) + "\n")
            f.write("process_action_list:" + str(process_action_list) + "\n")

        action_duration_file = os.path.join(settings.proj_root_path, "data", msgs.algo, "action_durations" + str(settings.i_run) + ".txt")
        with open(action_duration_file, "w") as f:
            f.write(str(env.take_action_list))




       
