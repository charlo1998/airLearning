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

def train(env, agent, checkpoint="C:/Users/charl/workspace/airlearning/airlearning-rl/data/A2C-B/model"):
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

    agent.save("C:/Users/charl/workspace/airlearning/airlearning-rl/data/DQN-B/model") #todo: automate the path

def test(env, agent, filepath):
    model = DQN.load(filepath)
    obs = env.reset()
    for i in range(settings.testing_nb_episodes_per_model):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)

if __name__ == "__main__":
    env, agent = setup()
    train()
