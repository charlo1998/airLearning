import random
import os
import subprocess
import numpy as np
import psutil
import json
import settings
import msgs
import matplotlib.pyplot as plt
import shutil
import pandas as pd
import subprocess
import ast
import math
import time

from tangent_bug import tangent_bug

def parse_data(file_name):
    with open(file_name  , 'a+') as f:
        f.seek(0,0)
        lines = f.readlines()
        if lines[-1] != '}':
            f.write("\n}")

    if (file_name == ''):
        file_hndl = open(os.path.join(settings.proj_root_path, "data", msgs.algo, msgs.mode + "_episodal_log.txt"), "r")
    else:
        file_hndl = open(file_name, "r")
    #print(f"parsing: {file_name}")
    data = json.load(file_hndl)
    data_clusterd_based_on_key = {}

    data.pop('difficulty', None) #removing settings data from the training data to be parsed
    data.pop('ClockSpeed', None)
    data.pop('loaded from checkpoint', None)

    for episode, episode_data in data.items(): #loops through the episode and get the keys (ex. success rate, n step, etc.)
        for key, value in episode_data.items(): #loops through all the keys of an episode and get their values
            if not (key in data_clusterd_based_on_key.keys()):
                data_clusterd_based_on_key[key] = [value]
            else:
                data_clusterd_based_on_key[key].append(value)

    return data_clusterd_based_on_key


def santize_data(file):
    tmp = "tmp.txt"
    f1 = open(file, 'r')
    f2 = open(tmp, 'w')
    for line in f1:
        for substring in (("True", '"' + str(True) + '"'), ("False", '"' + str(False) + '"')):
            line = line.replace(*substring)
        f2.write(line)
    f1.close()
    f2.close()
    shutil.move(tmp, file)

#def dict_mean(dict_list, data_to_inquire):
#    mean_dict = {}
#    keys_to_mean = []
#    for key in dict_list[0].keys():
#        mean_dict[key] = dict_list[0][key]
#    for el in data_to_inquire:
#        keys_to_mean.append(el[1])
#    for key in keys_to_mean:
#        #mean_dict[key] = np.mean([d[key] for d in dict_list], axis=0)
#        mean_dict[key] = [d[key] for d in dict_list]
#        print(mean_dict)
#    return mean_dict

def plot_trajectories(file):
    print("collecting trajectories")
    data = parse_data(file)
    nbOfEpisodesToPlot = 10
    assert(len(data['stepN']) >= 2*nbOfEpisodesToPlot)
    nbOfSteps = 0
    #plot the first x episodes
    plt.figure()
    for i in range(nbOfEpisodesToPlot): #this is the number of trajectories to plot
        xcoord = []
        ycoord = []
        episodeLength = data['stepN'][i]
        #converting string into list of floats
        coords = [x.strip('[]').split(' ') for x in data["position_in_each_step"][i][nbOfSteps:nbOfSteps+episodeLength]] #since the positions are appended, we need to remove the first nbOfSteps elements
        coords = [list(filter(None, x)) for x in coords]
        #print(len(coords))
        for coord in coords:
            positions = ([float(x) for x in coord])
            xcoord.append(positions[0])
            ycoord.append(positions[1])

        nbOfSteps += episodeLength
        plt.plot(xcoord, ycoord)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f'first {nbOfEpisodesToPlot} episodes')
    plt.xlim([-50, 50])
    plt.ylim([-50, 50])

    #plot the last 10 episodes
    nbOfSteps = data['total_step_count_for_experiment'][-nbOfEpisodesToPlot-1] #remove the steps before the last 10 episodes
    plt.figure()
    for i in range(-nbOfEpisodesToPlot,0): #this is the number of trajectories to plot
        xcoord = []
        ycoord = []
        episodeLength = data['stepN'][i]
        #converting string into list of floats
        coords = [x.strip('[]').split(' ') for x in data["position_in_each_step"][i][nbOfSteps:nbOfSteps+episodeLength]] #since the positions are appended, we need to remove the first nbOfSteps elements
        coords = [list(filter(None, x)) for x in coords]
        #print(len(coords))
        for coord in coords:
            positions = ([float(x) for x in coord])
            xcoord.append(positions[0])
            ycoord.append(positions[1])

        nbOfSteps += episodeLength
        plt.plot(xcoord, ycoord)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f'last {nbOfEpisodesToPlot} episodes')
    plt.xlim([-50, 50])
    plt.ylim([-50, 50])
    plt.show()

def average(data):
    #initializing list
    new_data = [[],[],[]]
    last_data_point = 0

    nb_of_data_points = 50
    nb_steps = data[0]["total_step_count_for_experiment"][-1]
    bucket_size = int(nb_steps/nb_of_data_points)
    if bucket_size < 1000:
        print("bucket size too small! verifiy settings.training_steps_cap")
        bucket_size = 1000
    i_step = 0
    while i_step < nb_steps:
        xbucket_avg = []
        ybucket_avg = []
        for k in range(settings.runs_to_do):
            #finding all values in current bucket
            xbucket = [x for x in data[k]["total_step_count_for_experiment"] if (i_step <= x <= i_step + bucket_size)]
            idx = [data[k]["total_step_count_for_experiment"].index(x) for x in xbucket]
            ybucket = [data[k]["total_reward"][i] for i in idx]
            #avg the bucket into 1 value
            xbucket = round(sum(xbucket)/len(xbucket)) if (len(xbucket) > 0) else i_step
            ybucket = round(sum(ybucket)/len(ybucket)) if (len(ybucket) > 0) else last_data_point
            #add the averagd value of all the runs into a list
            xbucket_avg.append(xbucket)
            ybucket_avg.append(ybucket)
        #add the avg of all the runs in a list
        new_data[0].append(round(sum(xbucket_avg)/len(xbucket_avg)))
        last_data_point = round(sum(ybucket_avg)/len(ybucket_avg))
        new_data[1].append(last_data_point)
        new_data[2].append(np.std(ybucket_avg, axis=0))

        i_step += bucket_size

    return new_data


def plot_data(file, data_to_inquire, mode="separate"):
    # santize_data(file)
    #make a list of dictionnaries, loop using setting.runs_to_do and average the values
    
    dataList = []
    plt.figure()
    for i in range(settings.runs_to_do):
        if settings.verbose:
            dataList.append(parse_data(file.replace("logverbose", "logverbose" + str(i))))
        else:
            dataList.append(parse_data(file.replace("log", "log" + str(i))))
            action_duration_file = os.path.join(settings.proj_root_path, "data", msgs.algo, "action_durations" + str(i) + ".txt")
            plot_histogram(action_duration_file)
    
    #print(dataList)
    data = dataList #to do, average the values instead of plotting them all. warning: the runs have different length of episodes!
    if settings.verbose:
        plot_sensor_usage(data)
    for el in data_to_inquire:
        plt.figure()
        if (el[0] == "total_step_count_for_experiment"):
            new_data = average(data)
            plt.plot(new_data[0], new_data[1])
            plt.fill_between(new_data[0], new_data[1] + np.array(new_data[2]), new_data[1] - np.array(new_data[2]), alpha=0.1)
            plt.title('averaged rewards as a function of the total timesteps')
        else:
            for i in range(settings.runs_to_do):
                #print(data[i][el[1]])
                plt.plot(data[i][el[0]], data[i][el[1]])
                assert (el[0] in data[i].keys())
        plt.xlabel(el[0])
        plt.ylabel(el[1])  
        plt.legend()
        plt.draw()
    if (mode == "separate"):
        plt.show()
    # plt.draw()
    # plt.pause(.001)


def generate_csv(file):
    for i in range(settings.runs_to_do):
        data = parse_data(file.replace("log", "log" + str(i)))
        data_frame = pd.DataFrame(data)
        data_frame.to_csv(file.replace("log", "log" + str(i)).replace("txt", "csv"), index=False)


def plot_sensor_usage(data):
    for k in range(settings.runs_to_do):
        nb_episodes = data[k]["episodeN"][-1]
        episode_actions = data[k]["actions_in_each_step"]
    sensors_per_action = []
    for i in range(nb_episodes):
        temp = []
        actions = episode_actions[i]
        #print(actions)
        for action in actions:
            action = action.replace(" ", ", ")
            #print(action)
            temp.append(np.sum(json.loads(action)))
        sensors_per_action.append(np.sum(temp)/len(temp))
    #print(sensors_per_action)
    plt.plot(range(len(sensors_per_action)),sensors_per_action)
    plt.xlabel('Number of episodes')
    plt.ylabel('number of sensors')
    plt.title('average number of sensors used in function of the episode during training')

def plot_histogram(file="C:/Users/Charles/workspace/airlearning/airlearning-rl/data/env/env_log.txt"):
    with open(file, 'r') as f:
        take_action_list = f.read()

    print("processing env_log.txt data")
    take_action_out = take_action_list.strip("[ ]\n") #removing brackets and spaces
    take_action_out = [float(value) for value in take_action_out.split(",")] #converting into list of floats

    n, bins, patches = plt.hist(take_action_out, bins = 'auto')
    plt.xlabel('action duration')
    plt.ylabel('frequency')
    maxfreq = n.max()
    plt.ylim(ymax=np.ceil(maxfreq/10)*10 if maxfreq % 10 else maxfreq + 10)


def append_log_file(episodeN, log_mode="verbose"):
    with open(os.path.join(settings.proj_root_path, "data", msgs.algo, msgs.mode + "_episodal_log" + log_mode + str(settings.i_run) + ".txt"),
              "a+") as f:
        if (episodeN == 0): #starting a new training task, add settings 
            f.write('{\n')
            f.write('"loaded from checkpoint":' + str(settings.use_checkpoint).lower() + ',\n')
            f.write('"difficulty":' + '"' + str(settings.difficulty) + '"' + ',\n')
            file_hndl = open(os.path.expanduser("~") + "/Documents/AirSim/settings.json", "r")
            #print(f"parsing: {file_name}")
            UnrealSettings = json.load(file_hndl)
            f.write('"ClockSpeed":' + str(UnrealSettings['ClockSpeed']) + ',\n')
        else:
            f.write(",\n")


        if (log_mode == "verbose"):
            f.write(
                '"' + str(episodeN) + '"' + ":" + str(msgs.episodal_log_dic_verbose).replace("\'", "\"").replace("True",
                                                                                                                 "\"True\"").replace(
                    "False", "\"False\""))
        # replace("\'", "\"") +",\n")
        else:
            f.write('"' + str(episodeN) + '"' + ":" + str(msgs.episodal_log_dic).replace("\'", "\"").replace("True",
                                                                                                             "\"True\"").replace(
                "False", "\"False\""))
        f.close()





def show_data_in_time():
    with open(settings.proj_root_path, "data", "DQN", "train_episodal_log.txt", "r") as f:
        data = json.load(f)
        print(data)


def airsimize_coordinates(pose):
    return [pose[0], pose[1], -pose[2]]


def list_diff(li1, li2):
    li_dif = [i for i in li1 + li2 if i not in li1 or i not in li2]
    return li_dif


def find_process_id_by_name(processName):
    '''
	Get a list of all the PIDs of a all the running process whose name contains
	the given string processName
	'''

    listOfProcessObjects = []

    # Iterate over the all the running process
    for proc in psutil.process_iter():
        try:
            pinfo = proc.as_dict(attrs=['pid', 'name'])
            # Check if process name contains the given name string.
            if processName.lower() in pinfo['name'].lower():
                listOfProcessObjects.append(pinfo['pid'])
        except Exception as e:
            pass

    return listOfProcessObjects;


def reset_msg_logs():
    # TODO this should be one by one cause then everytime we touch msgs we need to touch this as well
    msgs.success = False
    msgs.meta_data = {}
    msgs.episodal_log_dic = {}
    msgs.episodal_log_dic_verbose = {}
    msgs.cur_zone_number = 0
    msgs.weight_file_under_test = ''
    msgs.tst_inst_ctr = 0
    msgs.mode = ''
    msgs.restart_game_count = 0


def get_random_end_point(arena_size, split_index, total_num_of_splits):
    # distance from the walls
    wall_halo = floor_halo = roof_halo = 1
    goal_halo = settings.slow_down_activation_distance + 1

    sampling_quanta = .5  # sampling increment

    # how big the split is (in only one direction, i.e pos or neg)
    idx0_quanta = float((arena_size[0] - 2 * goal_halo - 2 * wall_halo)) / (2 * total_num_of_splits)
    idx1_quanta = float((arena_size[1] - 2 * goal_halo - 2 * wall_halo)) / (2 * total_num_of_splits)
    idx2_quanta = float((arena_size[2])) / (2 * total_num_of_splits)

    idx0_up_pos_bndry = (split_index + 1) * idx0_quanta
    idx1_up_pos_bndry = (split_index + 1) * idx1_quanta
    idx2_up_pos_bndry = (split_index + 1) * idx2_quanta

    #print(split_index)
    #print(total_num_of_splits)
    #print(idx0_up_pos_bndry)

    if (settings.end_randomization_mode == "inclusive"):
        idx0_low_pos_bndry = 3
        idx1_low_pos_bndry = 3
        idx2_low_pos_bndry = 0
    else:
        idx0_low_pos_bndry = (split_index) * idx0_quanta
        idx1_low_pos_bndry = (split_index) * idx1_quanta
        idx2_low_pos_bndry = (split_index) * idx2_quanta

    assert (
            idx0_up_pos_bndry - idx0_low_pos_bndry > sampling_quanta), "End doesn't fit within the zone, expand the arena size or reduce number of zones"
    assert (
            idx1_up_pos_bndry - idx1_low_pos_bndry > sampling_quanta), "End doesn't fit within the zone, expand the arena size or reduce number of zones"
    assert (
            idx2_up_pos_bndry - idx2_low_pos_bndry > sampling_quanta), "End doesn't fit within the zone, expand the arena size or reduce number of zones"

    rnd_pos_idx0 = random.choice(list(np.arange(
        idx0_low_pos_bndry + goal_halo, idx0_up_pos_bndry + goal_halo, sampling_quanta)))
    rnd_pos_idx1 = random.choice(list(np.arange(
        idx1_low_pos_bndry + goal_halo, idx1_up_pos_bndry + goal_halo, sampling_quanta)))
    rnd_pos_idx2 = random.choice(list(np.arange(
        idx2_low_pos_bndry + goal_halo, idx2_up_pos_bndry + goal_halo, sampling_quanta)))

    rnd_neg_idx0 = random.choice(list(np.arange(
        -idx0_up_pos_bndry - goal_halo, -idx0_low_pos_bndry - goal_halo, sampling_quanta)))

    rnd_neg_idx1 = random.choice(list(np.arange(
        -idx1_up_pos_bndry - goal_halo, -idx1_low_pos_bndry - goal_halo, sampling_quanta)))

    rnd_neg_idx2 = random.choice(list(np.arange(
        -idx2_up_pos_bndry - goal_halo, -idx2_low_pos_bndry - goal_halo, sampling_quanta)))

    rnd_idx0 = random.choice([rnd_neg_idx0, rnd_pos_idx0])
    rnd_idx1 = random.choice([rnd_neg_idx1, rnd_pos_idx1])
    rnd_idx2 = random.choice([rnd_neg_idx2, rnd_pos_idx2])

    """
	idx0_up_pos_bndry = int(arena_size[0]/2)
	idx1__up_pos_bndry = int(arena_size[1]/2)
	idx2__up_pos_bndry = int(arena_size[2])

	idx0_neg_bndry = int(-1*arena_size[0]/2)
	idx1_neg_bndry = int(-1*arena_size[1]/2)
	idx2_neg_bndry = 0
	
	rnd_idx0 = random.choice(list(range(
		idx0_neg_bndry + end_halo, idx0_pos_bndry - end_halo)))
	
	rnd_idx1 = random.choice(list(range(
		idx1_neg_bndry + end_halo, idx1_pos_bndry - end_halo)))
	 
	rnd_idx2 = random.choice(list(range(
	   0 + floor_halo, idx2_pos_bndry - roof_halo)))
	"""
    grounded_idx2 = 0  # to force the end on the ground, otherwise, it'll
    # be fallen (due to gravity) but then distance
    # calculation to goal becomes faulty

    if (rnd_idx0 == rnd_idx1 == 0):  # to avoid being on the start position
        rnd_idx0 = idx0_pos_bndry - end_halo

    #rnd_idx0 = 45
    #rnd_idx1 = 48
    #print(f"goal set to: {[rnd_idx0, rnd_idx1, grounded_idx2]} by get_random_en_point()")


    return [rnd_idx0, rnd_idx1, grounded_idx2]



def get_lib_addr():
    import rl.agents.dqn as blah
    # from shutil import copy2
    airsim_dir = os.path.dirname(blah.__file__)
    file_path = os.path.realpath(__file__)
    dir_path = os.path.dirname(file_path)
    print(airsim_dir)


def copy_json_to_server(filename):
    try:
        exit_code = os.system("copy " + filename + " " + settings.unreal_host_shared_dir)
        if not(exit_code == 0):
            raise Exception("couldn't copy the json file to the unreal_host_shared_dir")
    except Exception as e:
        print(str(e))
        exit(1)

class gofai():
    '''
    naive implementation of a pursuing algorithm with obstacle avoidance.
    '''

    def __init__(self):
        self.arc = 2*math.pi/settings.action_discretization #rad
        self.heading_coeff = 1
        self.safety_coeff = 3
        self.safety_dist = 1.5
        self.previous_obs = [3]*(settings.action_discretization+4)
        self.bug = tangent_bug()




    def predict(self, obs, goal):
        '''
        observation is in the form [angle, d_goal, y_vel, x_vel, d1, d2, ..., dn] where d1 starts at 180 deg and goes ccw, velocities are in drone's body frame ref
        actions are distributed as following:
        0-15: small circle
        16-31: medium small circle
        32-47: medium big circle
        48-63: big circle
        '''

        obs = obs[0][0] #flattening the list
        #obs[4:] = 100**obs[4:] #reconverting from normalized to real values
        #obs[1] = 100**obs[1]

        

        goal_angle = obs[0]*math.pi #rad
        global_goal_distance = obs[1]
        x_goal = goal[0]
        y_goal = goal[1]

        x_vel = obs[3]
        y_vel = obs[2]
        sensors = obs[4:]
        angles =  np.arange(-math.pi,math.pi,self.arc)

        objects =[]
        orientations = []
        #create objects list to evaluate obstacles positions, and replace missing values with old observations
        for i, sensor in enumerate(sensors):
            if sensor < 99:
                if sensor >= 66:
                    sensors[i] = self.previous_obs[i]
                objects.append(sensors[i])
                orientations.append(angles[i])
            

        #print(f"angle to goal: {goal_angle*180/math.pi}")
        #print(f"distance to goal: {global_goal_distance}")
        print(f"sensors: {np.round(sensors,1)}")
        
        
        #sensors = np.concatenate((sensors,sensors)) #this way we can more easily slice the angles we want
        #angles = np.concatenate((angles,angles))

        bestBenefit = -1000
        action = 0
        
        for i in range(settings.action_discretization*4): #settings.action_discretization*4
            theta = math.pi/2 - self.arc*(i%settings.action_discretization)  #in the action space, the circle starts at 90 deg and goes cw (drone body frame reference)
            #idx = 15 + 12 - i%settings.action_discretization
            #thetas = angles[idx-3:idx+5]

            #computing new distance to goal
            travel_dist = settings.base_speed*2**(i//settings.action_discretization)*(settings.mv_fw_dur) #travelled distance can be 0.5, 1, 2, or 4 times duration
            x_dest = travel_dist*math.cos(theta)*0.5 + x_vel * 0.75 # correcting for current speed since change in speed isn't instantaneous
            y_dest = travel_dist*math.sin(theta)*0.5 + y_vel * 0.75

            x_goal2 = global_goal_distance*math.sin(goal_angle) #reference frame for angle to goal is inverted
            y_goal2 = global_goal_distance*math.cos(goal_angle)

            new_dist = np.sqrt((x_goal-x_dest)**2+(y_goal-y_dest)**2)

            #computing the closest obstacle to the trajectory
            minDist = self.safety_dist
            if (len(objects) > 0):
                for object,angle in zip(objects,orientations):
                    x_obj = object*math.cos(angle+self.arc/2)
                    y_obj = object*math.sin(angle+self.arc/2)
                    dist = self.shortest_distance_on_trajectory(x_obj,y_obj,x_dest,y_dest)
                    if dist < minDist:
                        minDist = dist

            #computing the benefit
            benefit = self.heading_coeff*(global_goal_distance-new_dist) - self.safety_coeff*(self.safety_dist - minDist)
            #print(f"heading term: {global_goal_distance-new_dist}")
            #print(f"safety term: {self.safety_dist - minDist}")
            if benefit > bestBenefit:
                bestBenefit = benefit
                action =i


        self.previous_obs = sensors

        ### -----------printing info on the chosen action-------------------------------------------------------------
        print(f"desired angle: {np.round(theta*180/math.pi,1)}")
        #print(f"current speed: {[np.round(y_vel,1), np.round(x_vel,1)]}")
        #print(f"min distance in chosen trajectory: {np.round(minDist,5)}")
        #print(f"objects: {np.round(objects,1)}")
        #print(f"orientations: {np.round(orientations,2)}")
        #print(f"sensors: {np.round(sensors,1)}")
        #print(f"angles: {np.round(angles,2)}")
        print(f"received goal (relative): {[y_goal,x_goal]}")
        print(f"observed goal (relative): {[y_goal2,x_goal2]}")
        print(f"goal_distance: {global_goal_distance} angle: {goal_angle*180/math.pi}")
        #print(f"destination: {[np.round(y_dest,1), np.round(x_dest,1)]}")
        #print(f"destination: {np.round(now,2)}")
        #print(f"min distance in chosen trajectory: {minDist}")
        #print(f"predicted destination: {np.round(now,2)}")
        #print(f"new_dist: {new_dist}")
        #---------------------------------------------

        
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

