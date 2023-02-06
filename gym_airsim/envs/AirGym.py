import os ,sys

import logging
from settings_folder import settings
from game_config_handler_class import *
from game_handler_class import *
import file_handling
import msgs
#from common import utils
import json
import copy
import gym
from gym import spaces
from gym.utils import seeding
from algorithms.continuous.ddpg.OU import OU
import random
import time
from gym_airsim.envs.airlearningclient import *

from utils import append_log_file
from utils import gofai

logger = logging.getLogger(__name__)


def child_step(self, conn, airgym_obj):
        collided = False
        now = [0,0,0]
        track = 0.0
        old_depth = self.depth
        old_position = self.position
        old_velocity = self.velocity
        try:
            goal = [0,0,0]
            collided = airgym_obj.take_discrete_action(action)
            now = airgym_obj.drone_pos()
            track = airgym_obj.goal_direction(goal, now)
            self.depth = airgym_obj.getScreenDepthVis(track)
            self.position = airgym_obj.get_distance(goal)
            self.velocity = airgym_obj.drone_velocity()
            excp_occured = False
            conn.send([collided, now, track, self.depth, self.position, self.velocity, excp_occured])
            conn.close()
        except Exception as e:
            print(str(e) + "occured in child step")
            excp_occured = True

            conn.send([collided, now, track, old_depth, old_position, old_velocity, excp_occured])
            conn.close()



class AirSimEnv(gym.Env):
    #3.6.8 (self.airgym = None

    def __init__(self):
        # left depth, center depth, right depth, yaw
        if(settings.concatenate_inputs):
            if(settings.goal_position and settings.velocity): #for ablation studies
                STATE_POS = 2
                STATE_VEL = 2
            elif(settings.goal_position):
                STATE_POS = 2
                STATE_VEL = 0
            elif(settings.velocity):
                STATE_POS = 0
                STATE_VEL = 2
            else:
                STATE_POS = 0
                STATE_VEL = 0

            STATE_DISTANCES = settings.number_of_sensors
            if(msgs.algo == "SAC"):
                self.observation_space = spaces.Box(low=-1, high=1, shape=(( 1, STATE_POS + STATE_VEL + STATE_DEPTH_H * STATE_DEPTH_W)))
            else:
                self.observation_space = spaces.Box(low=-1, high=1,
                                                    shape=((1, STATE_POS + STATE_VEL + STATE_DISTANCES)))
        else:
            self.observation_space = spaces.Box(low=0, high=255, shape=(STATE_DISTANCES))

        self.total_step_count_for_experiment = 0 # self explanatory
        self.ease_ctr = 0  #counting how many times we ease the randomization and tightened it
        self.window_restart_ctr = 0 # counts the number of time we have restarted the window due to not meeting
                                    # the desired success_ratio
        if settings.profile:
            self.this_time = 0
            self.prev_time = 0
            self.loop_rate_list = []
            self.all_loop_rates = []
            self.take_action_list = []
            self.clct_state_list = []
            self.process_action_list = []

        self.episodeInWindow = 0
        self.passed_all_zones = False #whether we met all the zone success
        self.weight_file_name = '' #only used for testing
        self.log_dic = {}
        #self.cur_zone_number = 0
        self.cur_zone_number_buff = 0
        self.success_history = []
        self.success_ratio_within_window = 0
        self.episodeNInZone = 0 #counts the numbers of the episodes per Zone
                                #,hence gets reset upon moving on to new zone
        self.count = 0
        self.check_point = file_handling.CheckPoint()
        self.game_handler = GameHandler()
        self.OU = OU()
        self.game_config_handler = GameConfigHandler()
        if(settings.concatenate_inputs):
            self.concat_state = np.zeros((1, 1, STATE_POS + STATE_VEL + STATE_DISTANCES), dtype=np.uint8)
        self.depth = np.zeros((154, 256), dtype=np.uint8)
        self.rgb = np.zeros((154, 256, 3), dtype=np.uint8)
        self.grey = np.zeros((144, 256), dtype=np.uint8)
        self.position = np.zeros((2,), dtype=np.float32)
        self.velocity = np.zeros((3,), dtype=np.float32)
        self.distances = np.zeros(4)
        self.speed = 0
        self.track = 0
        self.collided = False
        self.prev_state = self.state()
        self.prev_info = {"x_pos": 0, "y_pos": 0}
        self.success = False
        self.zone = 0
        self.total_streched_ctr = 0

        self.actions_in_step = []
        self.position_in_step = []
        self.distance_in_step = []
        self.reward_in_step=[]
        self.total_reward = 0
        
        if(msgs.algo == "DDPG"):
            self.actor = ""
            self.critic= ""
        else:
            self.model = ""

        # pitch, yaw and roll are in radians ( min : -45 deg, max: 45 deg)
        if(msgs.algo == "DDPG"): #fixed typo, was "DDPG "
            self.action_space = spaces.Box(np.array([-0.785, -0.785, -0.785]),
                                       np.array([+0.785, +0.785, +0.785]),
                                       dtype=np.float32)  # pitch, roll, yaw_rate
        elif(msgs.algo == "PPO"):
            self.action_space = spaces.Box(np.array([-3.0, -3.0, -3.14]),
                                       np.array([+5.0, +5.0, 3.14]),
                                       dtype=np.float32)
        elif(msgs.algo == "SAC"):
            self.action_space = spaces.Box(np.array([-5.0, -5.0]),
                                       np.array([+5.0, +5.0]),
                                       dtype=np.float32)
        else:
            if(msgs.algo == "GOFAI"):
                #this is for baseline moving actions
                if(settings.timedActions or settings.positionActions):
                    self.nb_action_types = 4 
                    self.action_space = spaces.Discrete(self.nb_action_types * settings.action_discretization)
                else:
                    self.action_space = spaces.Discrete(20)
            else:
                #this is for RL on choosing observations
                self.action_space = spaces.MultiDiscrete([2]*settings.number_of_sensors) # one for each sensor 
                self.DWA = gofai()


        self.goal = utils.airsimize_coordinates(self.game_config_handler.get_cur_item("End"))
        self.episodeN = 0
        self.stepN = 0

        self.allLogs = {'reward': [0]}
        self.allLogs['distance'] = [float(np.sqrt(np.power((self.goal[0]), 2) + np.power(self.goal[1], 2)))]
        self.allLogs['track'] = [-2]
        self.allLogs['action'] = [1]

        self._seed()

        self.airgym = AirLearningClient()

    def set_model(self, model):
        self.model = model

    def set_actor_critic(self, actor, critic):
        self.actor = actor
        self.critic = critic
    
    # This function was introduced (instead of the body to be merged into
    # __init__ because I need difficulty level as an argument but I can't
    # touch __init__ number of arguments
    def setConfigHandlerRange(self, range_dic):
        self.game_config_handler.set_range(*[el for el in range_dic.items()])
        self.game_config_handler.populate_zones()

    """ 
    def set_test_vars(self, weight_file_name, test_instance_number):
        self.weight_file_name = weight_file_name
        self.test_instance_number = test_instance_number
    """

    def init_again(self, range_dic): #need this cause we can't pass arguments to
                                     # the main init function easily
        self.game_config_handler.set_range(*[el for el in range_dic.items()])
        self.game_config_handler.populate_zones()
        self.sampleGameConfig()
        self.goal = utils.airsimize_coordinates(self.game_config_handler.get_cur_item("End"))
    
    def setRangeAndSampleAndReset(self, range_dic):
        self.game_config_handler.set_range(*[el for el in range_dic.items()])
        self.game_config_handler.populate_zones()
        self.sampleGameConfig()
        self.goal = utils.airsimize_coordinates(self.game_config_handler.get_cur_item("End"))
        if(os.name=="nt"):
            self.airgym.unreal_reset()
        time.sleep(5)

    def getGoal(self): #there is no setting for the goal, cause you set goal
                   #indirectory by setting End
        return self.goal

    def state(self):
        if(msgs.algo == "DDPG"):
            return self.depth
        elif(msgs.algo == "PPO"):
            return self.concat_state
        elif(msgs.algo == "SAC"):
            return self.concat_state
        elif(msgs.algo == "DQN-B"):
            return self.concat_state
        elif(msgs.algo == "A2C-B"):
            return self.concat_state
        elif(msgs.algo == "GOFAI"):
            return self.concat_state
        else:
            return self.depth, self.velocity, self.position
    
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    #def computeReward(self, now):
    #    # test if getPosition works here liek that
    #    # get exact coordiantes of the tip
    #    distance_now = np.sqrt(np.power((self.goal[0] - now[0]), 2) + np.power((self.goal[1] - now[1]), 2))
    #    distance_before = self.allLogs['distance'][-1]
    #    distance_correction = (distance_before - distance_now)
    #    r = -1

    #    # check if you are too close to the goal, if yes, you need to reduce the yaw and speed
    #    if distance_now < settings.slow_down_activation_distance:
    #        yaw_correction =  abs(self.track) * distance_now 
    #        velocity_correction = (settings.mv_fw_spd_5 - self.speed)* settings.mv_fw_dur
    #        r = r + distance_correction + velocity_correction
    #    else:
    #        r = r + distance_correction
    #    return r, distance_now

    def computeReward(self, action):
        #if success ratio is more than 90%, try to penalize sensor usage. else, encourage more sensors for better performance.
        nb_sensors = np.sum(action)
        if len(self.success_history) > 0:
            success_ratio = float(sum(self.success_history)/len(self.success_history))
        else:
            success_ratio = 0

        if (success_ratio > 0.9):
            r = 16-nb_sensors
        else:
            r = nb_sensors-16

        #print(r)

        return r

    def ddpg_add_noise_action(self, actions):
        noise_t = np.zeros([1, self.action_space.shape[0]])
        a_t = np.zeros([1, self.action_space.shape[0]])
        noise_t[0][0] = max(settings.epsilon, 0) * self.OU.function(actions[0][0], 0.0, 0.60, 0.30)  # pitch
        noise_t[0][1] = max(settings.epsilon, 0) * self.OU.function(actions[0][1], 0.0, 0.60, 0.30)  # roll
        noise_t[0][2] = max(settings.epsilon, 0) * self.OU.function(actions[0][2], 0.0, 0.60, 0.30)  # yaw_rate

        if(random.random() < 0.1):
            print("********Now we apply the brake***********")
            noise_t[0][0] = max(settings.epsilon, 0) * self.OU.function(actions[0][0], 0.1, 1.00, 0.10)
            noise_t[0][1] = max(settings.epsilon, 0) * self.OU.function(actions[0][1], 0.1, 1.00, 0.10)
            noise_t[0][2] = max(settings.epsilon, 0) * self.OU.function(actions[0][2], 0.1, 1.00, 0.10)

        a_t[0][0] = (actions[0][0] + noise_t[0][0])
        a_t[0][1] = (actions[0][1] + noise_t[0][1])
        a_t[0][2] = (actions[0][2] + noise_t[0][2])
        throttle = settings.min_throttle + random.gauss(0.1, 0.05)
        duration = settings.duration
        actions_with_noise = [a_t[0][0], a_t[0][1], throttle, a_t[0][2], duration]

        return actions_with_noise

    def ppo_call_back_emulator(self):
            if (msgs.mode == 'train'):
                append_log_file(self.episodeN, "verbose")
                append_log_file(self.episodeN, "")
                if not(msgs.success):
                    return
                weight_file_name = self.check_point.find_file_to_check_point(msgs.cur_zone_number)
                self.model.save(weight_file_name)
                with open(weight_file_name+"_meta_data", "w") as file_hndle:
                    json.dump(msgs.meta_data, file_hndle)
            elif (msgs.mode == 'test'):
                append_log_file(self.episodeN, "verbose")
                append_log_file(self.episodeN, "")
                with open(msgs.weight_file_under_test+"_test"+str(msgs.tst_inst_ctr) + "_meta_data", "w") as file_hndle:
                    json.dump(msgs.meta_data, file_hndle)
            else:
                print("this mode " + str(msgs.mode) + "is not defined. only train and test defined")
                exit(0)
    def sac_call_back_emulator(self):
            if (msgs.mode == 'train'):
                append_log_file(self.episodeN, "verbose")
                append_log_file(self.episodeN, "")
                if not(msgs.success):
                    return
                weight_file_name = self.check_point.find_file_to_check_point(msgs.cur_zone_number)
                self.model.save(weight_file_name)
                with open(weight_file_name+"_meta_data", "w") as file_hndle:
                    json.dump(msgs.meta_data, file_hndle)
            elif (msgs.mode == 'test'):
                append_log_file(self.episodeN, "verbose")
                append_log_file(self.episodeN, "")
                with open(msgs.weight_file_under_test+"_test"+str(msgs.tst_inst_ctr) + "_meta_data", "w") as file_hndle:
                    json.dump(msgs.meta_data, file_hndle)
            else:
                print("this mode " + str(msgs.mode) + "is not defined. only train and test defined")
                exit(0)

    def ddpg_call_back_emulator(self): #should be called at the end of each episode
        if(msgs.algo=="DDPG"):
            if (msgs.mode == 'train'):
                append_log_file(self.episodeN, "verbose")
                append_log_file(self.episodeN, "")
                if not(msgs.success):
                    return
                weight_file_name = self.check_point.find_file_to_check_point(msgs.cur_zone_number)
                weight_file_name = weight_file_name.replace('_critic', '')
                self.actor.save_weights(weight_file_name+"_actor", overwrite=True)
                self.critic.save_weights(weight_file_name+"_critic", overwrite=True)
                with open(weight_file_name+"_meta_data", "w") as file_hndle:
                    json.dump(msgs.meta_data, file_hndle)
            elif (msgs.mode == 'test'):
                append_log_file(self.episodeN, "verbose")
                append_log_file(self.episodeN, "")
                with open(msgs.weight_file_under_test+"_test"+str(msgs.tst_inst_ctr) + "_meta_data", "w") as file_hndle:
                    json.dump(msgs.meta_data, file_hndle)
            else:
                print("this mode " + str(msgs.mode) + "is not defined. only train and test defined")
                exit(0)
        else:
            return
    def dqn_baselines_call_back_emulator(self):
            if (msgs.mode == 'train'):
                if (settings.verbose):
                    append_log_file(self.episodeN-1, "verbose")
                append_log_file(self.episodeN-1, "")
                if not(msgs.success):
                    return
                weight_file_name = self.check_point.find_file_to_check_point(msgs.cur_zone_number)
                weight_file_name = os.path.splitext(weight_file_name)[0]
                self.model.save(weight_file_name)
                with open(weight_file_name+"_meta_data", "w") as file_hndle:
                    json.dump(msgs.meta_data, file_hndle)
            elif (msgs.mode == 'test'):
                append_log_file(self.episodeN-1, "verbose")
                append_log_file(self.episodeN-1, "")
                with open(msgs.weight_file_under_test.replace('.pkl','') + "_test"+str(msgs.tst_inst_ctr) + "_meta_data.txt", "w") as file_hndle:
                    json.dump(msgs.meta_data, file_hndle)
                    json.dump(msgs.meta_data, file_hndle)
            else:
                print("this mode " + str(msgs.mode) + "is not defined. only train and test defined")
                exit(0)


    def update_success_rate(self):
        self.success_ratio_within_window = float(sum(self.success_history)/settings.update_zone_window)

    def update_zone_if_necessary(self):
        if (msgs.mode == 'train'):
            #TODO update_zone should be more general, i.e. called for other vars
            if self.success_rate_met() and not self.passed_all_zones:
                if (self.ease_ctr > 0):
                    self.tight_randomization()
                    return
                elif not(self.cur_zone_number_buff  == (settings.max_zone - 1)):
                    self.zone += 1
                    self.cur_zone_number_buff +=1
                else:
                    self.passed_all_zones = True
                    return
                self.start_new_window()
                self.update_zone("End")
        elif (msgs.mode == 'test'):
            if (self.episodeN % settings.testing_nb_episodes_per_zone == 0):
                if not(self.cur_zone_number_buff  == (settings.max_zone - 1)):
                    self.zone += 1
                    self.cur_zone_number_buff +=1
                self.update_zone("End")
        else:
            print("this mode " + str(msgs.mode) + "is not defined. only train and test defined")
            exit(0)


    def populate_episodal_log_dic(self):
        msgs.episodal_log_dic.clear()
        msgs.episodal_log_dic_verbose.clear()
        msgs.episodal_log_dic["cur_zone_number"] = msgs.cur_zone_number
        msgs.episodal_log_dic["success_ratio"] = round(float(sum(self.success_history)/len(self.success_history)),3)
        msgs.episodal_log_dic["success_history"] = sum(self.success_history)
        msgs.episodal_log_dic["success"] = msgs.success
        msgs.episodal_log_dic["stepN"] = self.stepN
        msgs.episodal_log_dic["episodeN"] = self.episodeN
        msgs.episodal_log_dic["episodeNInZone"] = self.episodeNInZone
        msgs.episodal_log_dic["episodeInWindow"] = self.episodeInWindow
        msgs.episodal_log_dic["ease_count"] = self.ease_ctr
        msgs.episodal_log_dic["total_streched_ctr"] = self.total_streched_ctr
        msgs.episodal_log_dic["restart_game_count"] = msgs.restart_game_count
        msgs.episodal_log_dic["total_reward"] = self.total_reward
        msgs.episodal_log_dic["total_step_count_for_experiment"] = self.total_step_count_for_experiment
        msgs.episodal_log_dic["goal"] = self.goal
        msgs.episodal_log_dic["distance_traveled"] = self.airgym.client.getMultirotorState().trip_stats.distance_traveled
        msgs.episodal_log_dic["energy_consumed"] = self.airgym.client.getMultirotorState().trip_stats.energy_consumed
        msgs.episodal_log_dic["flight_time"] = self.airgym.client.getMultirotorState().trip_stats.flight_time
        msgs.episodal_log_dic["time_stamp"] = self.airgym.client.getMultirotorState().timestamp
        msgs.episodal_log_dic["weight_file_under_test"] = msgs.weight_file_under_test

        #verbose
        msgs.episodal_log_dic_verbose = copy.deepcopy(msgs.episodal_log_dic)
        msgs.episodal_log_dic_verbose["reward_in_each_step"] = self.reward_in_step
        if (msgs.mode == "test"):
            msgs.episodal_log_dic_verbose["actions_in_each_step"] = self.actions_in_step
            msgs.episodal_log_dic_verbose["distance_in_each_step"] = self.distance_in_step
            msgs.episodal_log_dic_verbose["position_in_each_step"] = self.position_in_step
        elif (msgs.mode == "train"):
            msgs.episodal_log_dic_verbose["actions_in_each_step"] = self.actions_in_step
        else:
            raise Exception(msgs.mode + "is not supported as a mode")

    def possible_to_meet_success_rate(self): 
        #Computes what is the best success ratio if all the episodes in the current window are successes
        #if this ratio is inferior to the acceptable rate (ex. 50%), the window is restarted.
        if self.episodeInWindow < settings.update_zone_window:
            best_success_rate_can_achieve_now =  float(((settings.update_zone_window - self.episodeInWindow) +\
                                                        sum(self.success_history[-self.episodeInWindow:]))/settings.update_zone_window)
        else:
            best_success_rate_can_achieve_now = float(sum(self.success_history))

        acceptable_success_rate =  settings.acceptable_success_rate_to_update_zone
        if (best_success_rate_can_achieve_now < acceptable_success_rate):
            print("cannot reach acceptable success rate, resetting window.")
            return False
        else:
            return True

    def ease_randomization(self):
        for k, v in settings.environment_change_frequency.items():
            settings.environment_change_frequency[k] += settings.ease_constant
        self.ease_ctr += 1
        self.total_streched_ctr +=1

    def tight_randomization(self):
        for k, v in settings.environment_change_frequency.items():
            settings.environment_change_frequency[k] = max(
                settings.environment_change_frequency[k] - settings.ease_constant, 1)
        self.ease_ctr -=1
        self.total_streched_ctr +=1

    def start_new_window(self):
        print("Started new window")
        self.window_restart_ctr = 0
        self.episodeInWindow = 0

    def restart_cur_window(self):
        self.window_restart_ctr +=1
        print("Re-started current window")
        self.episodeInWindow = 0
        if (self.window_restart_ctr > settings.window_restart_ctr_threshold):
            self.window_restart_ctr = 0
            self.ease_randomization()

    def success_rate_met(self):
        acceptable_success_rate =  settings.acceptable_success_rate_to_update_zone
        return (self.success_ratio_within_window >= acceptable_success_rate)

    #check if possible to meet the success rate at all
    def restart_window_if_necessary(self):
        if not(self.possible_to_meet_success_rate()):
            self.restart_cur_window()

    def on_episode_end(self):
        self.update_history(self.success)
        self.update_success_rate()
        if(os.name=="nt"):
            msgs.meta_data = {**self.game_config_handler.cur_game_config.get_all_items()}
        self.populate_episodal_log_dic()

        if(msgs.algo == "DDPG"):
            self.ddpg_call_back_emulator()
        elif(msgs.algo == "PPO"):
            self.ppo_call_back_emulator()
        elif(msgs.algo == "SAC"):
            self.sac_call_back_emulator()
        elif(msgs.algo == "DQN-B" or msgs.algo == "A2C-B" or msgs.algo == "GOFAI"):
            self.dqn_baselines_call_back_emulator()

        self.restart_window_if_necessary()
        self.update_zone_if_necessary()

        self.actions_in_step = []
        self.distance_in_step = []
        self.reward_in_step = []
        self.position_in_step
        self.total_reward = 0

        self.allLogs['distance'] = [float(np.sqrt(np.power((self.goal[0]), 2) + np.power(self.goal[1], 2)))]


    def step(self, action): #changed from _step
        
		
        msgs.success = False 
        msgs.meta_data = {}


        try:
            #print("ENter Step"+str(self.stepN))
            #print(f"action taken: {action}")
            self.addToLog('action', action)
            self.stepN += 1
            self.total_step_count_for_experiment +=1

            if(settings.profile):
                    self.this_time = time.time()
                    if(self.stepN > 1):
                        self.loop_rate_list.append(self.this_time - self.prev_time)
                    self.prev_time = time.time()
                    take_action_start = time.perf_counter()

            #do action
            if(msgs.algo == "GOFAI"): #only do the baseline action
                if(settings.timedActions):
                    self.collided = self.airgym.take_timed_action(action)
                elif(settings.positionActions):
                    self.collided = self.airgym.take_position_action(action)
                else:
                    self.collided = self.airgym.take_discrete_action(action)
                self.actions_in_step.append(str(action))
            else:  #determine observation based on meta-action
                process_action_start = time.perf_counter()
                #action = action*0 +1 #artificially set all to 1
                obs = self.airgym.take_meta_action(action, self.prev_state)
                #determine move action based on DWA
                moveAction = self.DWA.predict(obs)
                process_action_end = time.perf_counter()
                if(settings.profile):
                    self.process_action_list.append(process_action_end - process_action_start)
                if(msgs.algo == "DDPG"):
                    self.actions_in_step.append([action[0], action[1], action[2]])
                    action = self.ddpg_add_noise_action(action)
                    self.collided = self.airgym.take_continious_action(float(action[0]), float(action[1]), float(action[2]), float(action[3]),
                                                     float(action[4]))
                elif(msgs.algo == "PPO"):
                    self.actions_in_step.append([action[0], action[1], action[2]])
                    self.collided = self.airgym.take_continious_action(action)
                elif(msgs.algo == "SAC"):
                    self.actions_in_step.append([action[0], action[1]])
                    self.collided = self.airgym.take_continious_action(action)
                else:
                    if(settings.timedActions):
                        self.collided = self.airgym.take_timed_action(moveAction)
                    elif(settings.positionActions):
                        self.collided = self.airgym.take_position_action(moveAction)
                    else:
                        self.collided = self.airgym.take_discrete_action(moveAction)
                    self.actions_in_step.append(str(action))
                
            
            if(settings.profile):
                    take_action_end = time.perf_counter()
                    self.take_action_list.append(take_action_end - take_action_start)
                    clct_state_start = time.time()
            
            now = self.airgym.drone_pos()
            self.track = self.airgym.goal_direction(self.goal, now)
            
            #get observation
            if(msgs.algo == "DQN-B" or msgs.algo == "SAC" or msgs.algo == "PPO" or msgs.algo == "A2C-B" or msgs.algo == "GOFAI"):
                self.concat_state = self.airgym.getConcatState(self.track, self.goal)
            elif(msgs.algo == "DQN" or msgs.algo == "DDPG"):
                self.depth = self.airgym.getScreenDepthVis(self.track)
            else:
                print("not an implemented algo")
                self.distances = self.airgym.get_laser_state()
                self.position = self.airgym.get_distance(self.goal)

            if(settings.profile):
                clct_state_end = time.time()
                self.clct_state_list.append(clct_state_end - clct_state_start)


            self.velocity = self.airgym.drone_velocity()
            self.speed = np.sqrt(self.velocity[0]**2 + self.velocity[1]**2 +self.velocity[2]**2)
            #print("Speed:"+str(self.speed))
            distance = np.sqrt(np.power((self.goal[0] - now[0]), 2) + np.power((self.goal[1] - now[1]), 2))
            #print(distance)
            
            print(f"current pose: {np.round(now,2)}")
            #print("-------------------------------------------------------------------------------------------------------")
            print(f"goal pose: {self.goal}")
            
            
            if distance < settings.success_distance_to_goal: #we found the goal: 1000ptso
                done = True
                print("-----------success, be happy!--------")
                self.success = True
                msgs.success = True
                #print(self.goal)
                #print(now)
                # Todo: Add code for landing drone (Airsim API)
                reward = 100
                #self.collect_data()
            elif self.stepN >= settings.nb_max_episodes_steps: #ran out of time/battery: 100pts (avoided collision)
                done = True
                print("-----------drone ran out of time!--------")
                reward = 0.0
                self.success = False
            elif self.collided == True: #we collided with something: between -1000 and -250, and worst if the collision appears sooner
                done = True
                print("------------drone collided!--------")
                #reward = min(-(1000.0-4*self.stepN), -500)
                reward = -200
                self.success = False
            elif (now[2] < -15): # Penalize for flying away too high
                done = True
                reward = -100
                self.success = False
            else: #not finished, compute reward like this: r = -1 + getting closer + flying slow when close (see def of computeReward)
                reward= self.computeReward(action)
                done = False
                self.success = False

            #Todo: penalize for more crazy and unstable actions

            self.allLogs['distance'] = [float(distance)]
            self.distance_in_step.append(distance)
            self.count +=1
            self.reward_in_step.append(reward)
            self.total_reward = sum(self.reward_in_step)
            self.position_in_step.append(str(now))
            info = {"x_pos":now[0], "y_pos":now[1]}

            state = self.state()
            #print(state)
            self.prev_state = state
            self.prev_info = info

            #self.on_step_end()
            if (done):
                self.on_episode_end()


            return state, reward, done, info
        except Exception as e:
            print("------------------------- step failed ----------------  with"\
                    , e , " error")
            self.game_handler.restart_game()
            self.airgym = AirLearningClient()
            return self.prev_state, 0, True, self.prev_info

    def update_history(self, result):
        if (len(self.success_history) < settings.update_zone_window):
            self.success_history.append(result)
        else:
            self.success_history.pop(0)
            self.success_history.append(result)

    def addToLog(self, key, value):
        if key not in self.allLogs:
            self.allLogs[key] = []
        self.allLogs[key].append(value)

    def on_episode_start(self):
        if self.episodeN == 0:
            settings.i_run -= 1 #reduce run counter
        self.stepN = 0
        self.episodeN += 1
        self.episodeNInZone +=1
        #self.episodeInWindow = self.episodeNInZone % settings.update_zone_window
        self.episodeInWindow +=1
        now = self.airgym.drone_pos()
        self.track = self.airgym.goal_direction(self.goal, now)
        self.concat_state = self.airgym.getConcatState(self.track, self.goal)
        #self.depth = self.airgym.getScreenDepthVis(self.track)
        #self.rgb = self.airgym.getScreenRGB()
        self.position = self.airgym.get_distance(self.goal)
        self.velocity = self.airgym.drone_velocity()
        msgs.cur_zone_number = self.cur_zone_number_buff  #which delays the update for cur_zone_number

    def reset(self): #was_reset, which means it wasn't imported with a import * (don't know where it was imported yet
        print(f"finished episode {self.episodeN}, total step count: {self.total_step_count_for_experiment}")
        first = time.time()
        try:
            if(settings.profile):
                if(self.stepN % 20 ==0):
                    print("Average Loop Rate:"+str(np.mean(self.loop_rate_list)))
                    print ("Action Time:" +str(np.mean(self.take_action_list)))
                    print("Collect State Time"+str(np.mean(self.clct_state_list)))

            if(not self.collided): #no need to reset entire simulator
                #print("enter reset")
                vars_to_randomize = ["End"]
                self.sampleGameConfig(*vars_to_randomize) #sample a new End position with arena range
                self.goal = utils.airsimize_coordinates(self.game_config_handler.get_cur_item("End")) #set goal to new End position
                scnd = time.time()
                print(f"randomize_env: {np.round((scnd-first)*1000)} ms")
            else:
                self.randomize_env()
                if(os.name=="nt"):
                    connection_established = self.airgym.unreal_reset()
                    if not connection_established:
                        raise Exception
                time.sleep(2)
                self.airgym.AirSim_reset()
                scnd = time.time()
                print(f"done AirSim resetting: {np.round((scnd-first)*1000)} ms")


            self.on_episode_start()
            state = self.state()
            self.prev_state = state

            return state

        except Exception as e:
            print("------------------------- reset failed ----------------  with"\
                    , e , " error")
            self.game_handler.restart_game()
            self.airgym = AirLearningClient()
            self.on_episode_start()
            print("done on episode start")
            return self.prev_state



    def update_zone(self, *args):
        #all_keys = self.game_config_handler.game_config_range.find_all_keys()
        print("Zone updated! resetting success history")
        self.success_history = []
        self.game_config_handler.update_zone(*args)
        self.episodeNInZone = 0

    # generate new random environement if needed
    def randomize_env(self):
        vars_to_randomize = []
        for k, v in settings.environment_change_frequency.items(): #get the variable/frequency pairs
            
            if (self.episodeN+1) %  v == 0: #if they are due to randomize, pass them to the sample function
                vars_to_randomize.append(k)

        if (len(vars_to_randomize) > 0):
            self.sampleGameConfig(*vars_to_randomize)
            self.goal = utils.airsimize_coordinates(self.game_config_handler.get_cur_item("End"))

    def updateJson(self, *args):
        self.game_config_handler.update_json(*args)

    def getItemCurGameConfig(self, key):
        return self.game_config_handler.get_cur_item(key)

    def setRangeGameConfig(self, *args):
        self.game_config_handler.set_range(*args)

    def getRangeGameConfig(self, key):
        return self.game_config_handler.get_range(key)

    def sampleGameConfig(self, *arg):
        self.game_config_handler.sample(*arg)
