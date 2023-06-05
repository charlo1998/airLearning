import os
import psutil

os.sys.path.insert(0, os.path.abspath('../settings_folder'))


import gym
import time
import json
import tensorflow as tf
import settings
import utils
from tangent_bug import tangent_bug
import msgs
from gym_airsim.envs.airlearningclient import *
import callbacks
from multi_modal_policy import MultiInputPolicy
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.policies import MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
#from stable_baselines.common import make_vec_env #this yields an error
from stable_baselines import A2C
from customPolicy import CustomLSTMPolicy, CustomPolicy

#set high priority for process
p = psutil.Process(os.getpid())
p.nice(psutil.REALTIME_PRIORITY_CLASS)


modelpath = os.path.expanduser("~") + "/workspace/airlearning/airlearning-rl/data/A2C-B/model.pkl"
msgs.mode = 'test'
msgs.weight_file_under_test = modelpath

model = A2C.load(modelpath)
DWA = utils.gofai()
bug = tangent_bug()

#### ----------- profiling -------------------------------
#load series of observations (multiple episodes) and process them to be inputed to the agent
file = "C:/Users/charl/workspace/airlearning/airlearning-rl/data/A2C-B/test_episodal_logverbose.txt"
dataList = []
for i in range(settings.runs_to_do):
    if settings.verbose:
        dataList.append(utils.parse_data(file.replace("logverbose", "logverbose" + str(i))))
    else:
        dataList.append(utils.parse_data(file.replace("log", "log" + str(i))))
observations_episodes = dataList[0]["observations_in_each_step"]
states = []
sensors = []
for observations in observations_episodes:
    for observation in observations:
        observation = observation.replace("\n  ", " ")
        obs = np.array(json.loads(observation))
        obs = utils.normalize(obs)
        #print(observation)
        sensors.append(obs[6:settings.number_of_sensors+6])
        obs_shape = obs.shape
        obs = obs.reshape(1, obs_shape[0])
        obs = np.expand_dims(obs, axis=0)
        states.append(obs)

#gpu warmup
for observation in states:
    _ = model.predict(observation)

#run inference on all of them
actions = []
wall_start = time.perf_counter()
start = time.process_time()
for observation in states:
    actions.append(model.predict(observation))
total_inference = time.process_time() - start
wall_time_inference = time.perf_counter() - wall_start

#get the dummy observations for bug
goals = []
for observation in states:
    goals.append(bug.predict(observation))

#get the dummy observations for dwa
dwa_inputs = []
for i, action in enumerate(actions):
    action = action[0][0]
    #print(action)
    chosen_idx = np.argpartition(action, -settings.k_sensors)[-settings.k_sensors:]
    sensor_output = np.ones(settings.number_of_sensors)*100
    for idx in chosen_idx:
        sensor_score = action[idx]
        if (sensor_score >= 0.5):
            sensor_output[idx] = sensors[i][idx]
    dwa_inputs.append([[sensor_output]])  #to be consistent with the normal rl pipeline

#run dwa on all the model's outputs
wall_start = time.perf_counter()
start = time.process_time()
for i, obs in enumerate(dwa_inputs):
    _ = DWA.predict(obs,goals[i])
total_processing = time.process_time() - start
total_dwa_wall_time = time.perf_counter() - wall_start

print(f"avg inference CPU time: {total_inference/len(states)*1000} ms")
print(f"avg inference wall time: {wall_time_inference/len(states)*1000} ms")
print(f"avg dwa CPU time: {total_processing/len(states)*1000} ms")
print(f"avg dwa wall time: {total_dwa_wall_time/len(states)*1000} ms")

