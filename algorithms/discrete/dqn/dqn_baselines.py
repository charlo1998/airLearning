import sys
import gym

import os
import tensorflow as tf
os.sys.path.insert(0, os.path.abspath('../../../settings_folder'))
import settings
import msgs
from gym_airsim.envs.airlearningclient import *
import callbacks
from multi_modal_policy import MultiInputPolicy
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq import DQN, MlpPolicy
from stable_baselines.deepq.policies import MultiInputPolicy


from keras.backend.tensorflow_backend import set_session
from customPolicy import CustomLSTMPolicy, CustomPolicy

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
    agent = DQN(MlpPolicy, vec_env, verbose=1)
    env.set_model(agent)

    return env, agent

def train(env, agent, checkpoint=os.path.expanduser("~") + "/workspace/airlearning/airlearning-rl/data/DQN-B/model.pkl"):
    if settings.use_checkpoint:
        print(f"loading checkpoint {checkpoint}")
        agent = DQN.load(checkpoint)
        agent.env = DummyVecEnv([lambda: env])
    # Train the agent
    agent.learn(total_timesteps=settings.training_steps_cap)

    #env loop rate logging
    if settings.profile:
        with open(os.path.join(settings.proj_root_path, "data", "env","env_log.txt"),
            "w") as f:
            f.write("loop_rate_list:" + str(env.loop_rate_list) + "\n")
            f.write("take_action_list:" + str(env.take_action_list) + "\n")
            f.write("clct_state_list:" + str(env.clct_state_list) + "\n")
            f.write("process_action_list:" + str(env.process_action_list) + "\n")

        action_duration_file = os.path.join(settings.proj_root_path, "data", msgs.algo, "action_durations" + str(settings.i_run) + ".txt")
        with open(action_duration_file, "w") as f:
            f.write(str(env.take_action_list))

    agent.save(os.path.expanduser("~") + "/workspace/airlearning/airlearning-rl/data/DQN-B/model") #todo: automate the path

def test(env, agent, filepath):
    msgs.mode = 'test'
    msgs.weight_file_under_test = filepath

    model = DQN.load(filepath)
    
    for i in range(settings.testing_nb_episodes_per_model):
        obs = env.reset()
        done = False
        while not done:
            infer_start = time.perf_counter()
            cpu_start = time.process_time()
            action, _states = model.predict(obs)
            cpu_end = time.process_time()
            infer_end = time.perf_counter()
            infer_latency_list.append(infer_end-infer_start)
            infer_cpu_list.append(cpu_end-cpu_start)
            

            obs, rewards, done, info = env.step(action)
            #env.airgym.client.simPause(True)
            #answer = input()
            #env.airgym.client.simPause(False)
            
    print(f"total CPU processing time: {(time.process_time()-start)} s")
    print(f"average DWA clock processing time: {sum(env.process_action_list)/len(env.process_action_list)*1000} ms")
    print(f"average inference CPU processing time: {sum(infer_cpu_list)/len(infer_cpu_list)*1000} ms")

    #env loop rate logging
    if settings.profile:
        with open(os.path.join(settings.proj_root_path, "data", "env","env_log.txt"),
            "w") as f:
            f.write("loop_rate_list:" + str(env.loop_rate_list) + "\n")
            f.write("take_action_list:" + str(env.take_action_list) + "\n")
            f.write("clct_state_list:" + str(env.clct_state_list) + "\n")
            f.write("process_action_list:" + str(env.process_action_list) + "\n")

        inference_duration_file = os.path.join(settings.proj_root_path, "data", msgs.algo, "inference_durations" + str(settings.i_run) + ".txt")
        with open(inference_duration_file, "w") as f:
            f.write(str(infer_latency_list[1:]))

if __name__ == "__main__":
    env, agent = setup()
    train()
