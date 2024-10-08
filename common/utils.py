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
from bisect import bisect

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

def normalize(obs):
    """ takes an observation (list) and outputs the normalized form"""
    #print(obs)
    obs[6:settings.number_of_sensors+6] = np.log(obs[6:settings.number_of_sensors+6]+0.0001)/np.log(100) #this way gives more range to the smaller distances (large distances are less important).
    #obs[settings.number_of_sensors+6:] = obs[settings.number_of_sensors+6:]*np.pi
    obs[1] = np.log(obs[1]+0.0001)/np.log(100)
    obs[0] = obs[0]/np.pi #rad
    obs[2] = obs[2]/(settings.base_speed*20.0) #reconverting from normalized values
    obs[3] = obs[3]/np.pi
    obs[4:6] = obs[4:6]/50.0 

    return obs



def plot_trajectories(file):
    print("collecting trajectories")
    data = parse_data(file)
    nbOfEpisodesToPlot = min(50,len(data['stepN']))
    nbOfSteps = 0
    #plot the first x episodes
    plt.figure()
    xgoal = []
    ygoal = []
    episodes_to_plot = [0,1,2,3,4]
    for i in episodes_to_plot: #nbOfEpisodesToPlot this is the number of trajectories to plot
        xgoal.append(data['goal'][i][0])
        ygoal.append(data['goal'][i][1])
        xcoord = []
        ycoord = []
        episodeLength = data['stepN'][i]
        #converting string into list of floats
        coords = data["position_in_each_step"][i]
        
        #print(len(coords))
        for coord in coords:
            positions = ([float(x) for x in coord])
            xcoord.append(positions[0])
            ycoord.append(positions[1])

        nbOfSteps += episodeLength
        plt.plot(xcoord, ycoord, label=str(i))
        trajectories = np.array([xcoord, ycoord])
        np.savetxt(os.path.join(settings.proj_root_path, "data", msgs.algo,f"trajectories{i}.txt"), trajectories)

    plt.scatter(xgoal, ygoal)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f'first {nbOfEpisodesToPlot} episodes')
    arena_size = settings.hard_range_dic['ArenaSize'][0]
    print(arena_size)
    plt.xlim([-arena_size[0]/2, arena_size[0]/2])
    plt.ylim([-0, arena_size[1]/2])
    plt.legend()


    #plot distance traveled vs birdview distance to goal and mission time
    goal = data['goal'][0]
    collisions = 0
    fails = 0
    if data['success'][0] == "False": #only register the successful runs (running out of time increases distance traveled way to much, collision makes it too small)
        mission_times = [0]
        traveled_distances = [0]
        ideal_distances = [0]
        if data['stepN'][0] == settings.nb_max_episodes_steps:
            fails += 1
        else:
            collisions += 1
    else:
        ideal_distances = [np.sqrt(goal[1]**2+goal[0]**2)-settings.success_distance_to_goal]
        traveled_distances = [data['distance_traveled'][0]]
        mission_times = [data['flight_time'][0]]
    
    for i in range(1,len(data['distance_traveled'])):
        if data['success'][i] == "False": #only register the successful runs (running out of time increases distance traveled way to much, collision makes it too small)
            
            if data['stepN'][i] == settings.nb_max_episodes_steps:
                fails += 1
                mission_times.append(0)
                traveled_distances.append(0)
                ideal_distances.append(0)
            else:
                collisions += 1
                mission_times.append(-1)
                traveled_distances.append(-1)
                ideal_distances.append(-1)
            continue

        start = data['position_in_each_step'][i][0]
        end = data['goal'][i]
        if (data['success'][i-1] == "False" and data['stepN'][i-1] != settings.nb_max_episodes_steps) or i%50 == 0 or data['distance_traveled'][i-1] > data['distance_traveled'][i]: #the sim is reset to 0 after a crash or after every 50 episodes
                traveled_distances.append(data['distance_traveled'][i])
                ideal_distances.append(np.sqrt(end[1]**2+end[0]**2)-settings.success_distance_to_goal)
                mission_times.append(data['flight_time'][i])
        else:
            delta = data['distance_traveled'][i] - data['distance_traveled'][i-1]
            traveled_distances.append(delta)
            ideal_distances.append(np.sqrt((end[1]-start[1])**2+(end[0]-start[0])**2)-settings.success_distance_to_goal)
            mission_times.append(data['flight_time'][i] - data['flight_time'][i-1])

    print(f"There was {collisions/len(data['goal'])*100.0}% collisions and {fails/len(data['goal'])*100.0}% fails to reach the goal")
    with open(os.path.join(settings.proj_root_path, "data", msgs.algo, "metrics.txt"),
            "w") as f:
            f.write("traveled_distances:" + str(traveled_distances) + "\n")
            f.write("ideal_distances:" + str(ideal_distances) + "\n")
            f.write("mission_times:" + str(mission_times) + "\n")

    plt.figure()
    #plt.plot(range(nbOfEpisodesToPlot), ideal_distances, range(nbOfEpisodesToPlot), traveled_distances)
    ratio = [traveled_distance/ideal_distance if ideal_distance > 0 else 0 for (traveled_distance, ideal_distance) in zip(traveled_distances, ideal_distances)]
    n, bins, patches = plt.hist(ratio, bins = 'auto')
    plt.xlabel("traveled distance/birdview distance ratio")
    plt.ylabel("frequency")
    if  data['success_ratio'][-1] > 0.0:
        print(f"Average ratio of traveled distance/bird view distance: {sum(ratio)/(len(ratio)-fails-collisions)}")
        print(f"Average mission time: {sum([time if time > 0 else 0 for time in mission_times])/(len(mission_times)-fails-collisions)}")
    plt.figure()
    n, bins, patches = plt.hist(mission_times, bins = 'auto')
    plt.xlabel("mission length (s)")
    plt.ylabel("frequency")

    plt.show()



def average(data, key):
    """ returns the data averaged into bucket for more clarity. it also averages for all the runs"""
    #initializing list
    new_data = [[],[],[]]
    last_data_point = 0
    print(f"averaging data for {key}")

    nb_of_data_points = 250
    nb_steps = data[0]["total_step_count_for_experiment"][-1]
    bucket_size = int(nb_steps/nb_of_data_points)
    if bucket_size < settings.nb_max_episodes_steps:
        print("bucket size too small! verifiy settings.training_steps_cap")
        bucket_size = settings.nb_max_episodes_steps
    i_step = 0
    while i_step < nb_steps:
        xbucket_avg = []
        ybucket_avg = []
        for k in range(settings.runs_to_do):
            #finding all values in current bucket
            xbucket = [x for x in data[k]["total_step_count_for_experiment"] if (i_step <= x <= i_step + bucket_size)]
            idx = [data[k]["total_step_count_for_experiment"].index(x) for x in xbucket]
            ybucket = [data[k][key][i] for i in idx]
            if len(ybucket) == 0:
                print("empty bucket while averaging over multiple runs!")
            #avg the bucket into 1 value
            xbucket = round(sum(xbucket)/len(xbucket)) if (len(xbucket) > 0) else i_step
            ybucket = sum(ybucket)/len(ybucket) if (len(ybucket) > 0) else last_data_point
            #add the averagd value of all the runs into a list
            xbucket_avg.append(xbucket)
            ybucket_avg.append(ybucket)
        #add the avg of all the runs in a list
        new_data[0].append(round(sum(xbucket_avg)/len(xbucket_avg)))
        last_data_point = sum(ybucket_avg)/len(ybucket_avg)
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
        if msgs.algo != "GOFAI" and msgs.mode == "test":
            infer_duration_file = os.path.join(settings.proj_root_path, "data", msgs.algo, "inference_durations" + str(i) + ".txt")
            plot_histogram(infer_duration_file)
            plt.title('Distribution of the inference duration (latency) of the policy')
            plt.show()
    
    #print(dataList)
    data = dataList #to do, average the values instead of plotting them all. warning: the runs have different length of episodes!
    if settings.verbose:
        if msgs.algo != "GOFAI":
            plot_sensor_usage(data)
            if settings.visualize_actions:
                plot_action_vs_obs(data)
    for el in data_to_inquire:
        plt.figure()
        if settings.average_runs and el[0] == "total_step_count_for_experiment":
            new_data = average(data, el[1])
            if el[1] == "success_ratio":
                 np.savetxt(os.path.join(settings.proj_root_path, "data", msgs.algo,"SuccessRatio0.txt"), new_data)
            plt.plot(np.array(new_data[0])*3, new_data[1])
            plt.fill_between(np.array(new_data[0])*3, new_data[1] + np.array(new_data[2]), new_data[1] - np.array(new_data[2]), alpha=0.5)
        else:
            for i in range(settings.runs_to_do):
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

def plot_action_vs_obs(data):
    for k in range(settings.runs_to_do):
        episode_actions = data[k]["actions_in_each_step"]
        episode_observations = data[k]["observations_in_each_step"]
        dwa_actions = data[k]["planner_action_in_each_step"]
        if msgs.mode == "test":
            episode_positions = data[k]["position_in_each_step"]
            goals = data[k]["goal"]

    sensors_per_action = []
    obs_per_action = []
    pos_per_action = []
    dwa_goal_per_action = []
    for i, actions in enumerate(episode_actions):
        temp = []
        #print(actions)
        for action in actions:
            #print(action)
            temp.append(action)
        sensors_per_action.append(temp)

    for i, observations in enumerate(episode_observations):
        temp = []
        predicted = []
        #print(observations)
        for observation, dwa_action in zip(observations,dwa_actions[i]):
            #reading observation
            observation = observation.replace("\n  ", " ")
            #print(observation)
            obs = json.loads(observation)
            temp.append(obs[6:settings.number_of_sensors+6])
            #processing dwa action
            theta = math.pi/2 - (2*math.pi/settings.action_discretization)*(dwa_action%settings.action_discretization)  #in the action space, the circle starts at 90 deg and goes cw (drone body frame reference)
            travel_speed = min(2, settings.base_speed*3**(dwa_action//settings.action_discretization)) #travelling speed can be 0.1, 0.3, 0.9, or 2 
            vel_angle = obs[3]
            vel_norm = obs[2]
            x_dest = travel_speed*math.cos(theta)*0.4*(settings.mv_fw_dur) + vel_norm*math.cos(vel_angle)*0.75 # correcting for current speed since change in speed isn't instantaneous
            y_dest = travel_speed*math.sin(theta)*0.4*(settings.mv_fw_dur) + vel_norm*math.sin(vel_angle)*0.75
            predicted_angle = math.atan2(y_dest,x_dest)
            predicted_distance = np.sqrt(y_dest**2 + x_dest**2)
            predicted.append([predicted_angle, predicted_distance])

        dwa_goal_per_action.append(predicted)
        obs_per_action.append(temp) #appending to all observations of episode to list of episodes

    if msgs.mode == "test":
        for i, positions in enumerate(episode_positions):
            temp = []
            for position in positions:
                temp.append(position[0:2])
            pos_per_action.append(temp)
    

    r = np.arange(0, settings.number_of_sensors)
    theta = 2 * np.pi * (r+0.5) / settings.number_of_sensors - np.pi
    r2 = np.arange(0, 2*settings.number_of_sensors)
    theta2 = 2 * np.pi * (r2+1) / (2*settings.number_of_sensors) - np.pi
    
    

    #print(sensors_per_action[0])
    
    number_of_episodes_to_show = min(1, len(episode_actions))
    for episode in range(3,4):
        plt.figure()
        number_of_sensors = []
        for sensors in sensors_per_action[episode]:
            number_of_sensors.append(sum(sensors))
        plt.step(range(len(episode_actions[episode])), number_of_sensors)
        plt.xlabel('Time step')
        plt.ylabel('number of selected sectors')
        
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(2.1, 2.5))
        #ax.set_rmax(12)
        ax.set_ylim([0,12])
        #ax.set_rscale('symlog')
        #ax.set_title("sensor observation", va='bottom')
        
        ax.grid(True)

        for step in range(len(episode_actions[episode])):
            chosen_areas = [0]*2*settings.number_of_sensors
            for i, sensor in enumerate(sensors_per_action[episode][step]):
                if sensor:
                    chosen_areas[2*i-1] = 12
                    chosen_areas[2*i] = 12
                    chosen_areas[2*i+1] = 12

            #ax.set_rmax(12)
            ax.set_ylim([0,12])
            #ax.set_rscale('symlog')
            #ax.set_title("sensor observation for timestep " + str(step), va='bottom')
            ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
            line1 = ax.plot(list(theta) + [theta[0]], obs_per_action[episode][step] + [obs_per_action[episode][step][0]])
            line2 = plt.fill_between(theta2, 0, chosen_areas, alpha=0.2)
            if msgs.mode == "test":
                x_goal_rel = goals[episode][1]-pos_per_action[episode][step][1] #uav frame of ref
                y_goal_rel = goals[episode][0]-pos_per_action[episode][step][0]
                goal_norm = np.sqrt(x_goal_rel**2+y_goal_rel**2)
                goal_angle = math.atan2(y_goal_rel,x_goal_rel)
                line3 = ax.scatter(goal_angle, goal_norm, c= 'r')
                wanted_angle = dwa_goal_per_action[episode][step][0]
                wanted_norm = dwa_goal_per_action[episode][step][1]
                line3 = ax.scatter(wanted_angle, wanted_norm, c= 'g')
            ax.set_rticks([3, 6, 9])  # Less radial ticks
            ax.set_rlabel_position(180)  # Move radial labels away from plotted line
            selected_actions = [176]
            if(step == len(episode_actions[episode])-1): #pause longer for last step of the episode
                plt.pause(2)
            elif (step in selected_actions):
                plt.pause(0.6)
            else:
                plt.pause(0.03)
            plt.cla()
    plt.show()


def plot_sensor_usage(data):
    sensors_per_action = [[6] for i in range(settings.runs_to_do)]
    min_length = 9999999
    for k in range(settings.runs_to_do):
        episode_actions = data[k]["actions_in_each_step"]
        #print(len(episode_actions))
        window_size = max(1,round((len(episode_actions)*200/settings.testing_nb_episodes_per_model)))
        
        stds = []
        for actions in episode_actions:
            temp = []
            #print(actions)
            for action in actions:
                #print(action)
                sensors_per_action[k].append(np.sum(action))
        if len(sensors_per_action[k]) < min_length:
            min_length = len(sensors_per_action[k])
    
    for k in range(settings.runs_to_do): #slicing to the smallest dataset
        sensors_per_action[k] = sensors_per_action[k][0:min_length]

    avg_sensors_per_action = np.array(np.sum(sensors_per_action, axis=0))/settings.runs_to_do
    std = np.std(sensors_per_action, axis=0)
    smoothed = np.convolve(avg_sensors_per_action, np.ones(window_size)/window_size, mode='valid')
    smoothed_std = np.convolve(std, np.ones(window_size)/window_size, mode='valid')*0.5
    #std = np.std(np.array([avg_sensors_per_action[i:i+window_size] for i in range(len(avg_sensors_per_action)-window_size+1)]), axis=1)
    print(f"total average sensors_per_action: {sum(avg_sensors_per_action)/len(avg_sensors_per_action)}")
    print(f"sensor usage window size: {window_size}")
    plt.plot(np.array(range(len(smoothed)))*3, smoothed)
    plt.fill_between(np.array(range(len(smoothed)))*3, smoothed-smoothed_std, smoothed+smoothed_std, alpha=0.5)
    plt.xlabel('Time step')
    plt.ylabel('number of sensors')
    plt.title('average number of sensors used in the timesteps during training')

    data = np.vstack((avg_sensors_per_action, std))
    np.savetxt(os.path.join(settings.proj_root_path, "data", msgs.algo,"sensors0.txt"), data)

def plot_histogram(file="C:/Users/Charles/workspace/airlearning/airlearning-rl/data/env/env_log.txt"):
    with open(file, 'r') as f:
        take_action_list = f.read()

    print("processing data")
    take_action_out = take_action_list.strip("[ ]\n") #removing brackets and spaces
    take_action_out = [float(value) for value in take_action_out.split(",")] #converting into list of floats
    print(f"average inference time: {sum(take_action_out)/len(take_action_out)*1000} ms")

    n, bins, patches = plt.hist(take_action_out, bins = 'auto')
    plt.xlabel('Duration (s)')
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

    if settings.deterministic:
        [rnd_idx0, rnd_idx1, grounded_idx2] = settings.goals_list[settings.goals_idx]
        #settings.goals_idx +=1 
        print(f"goal set to: {[rnd_idx0, rnd_idx1]}")
        return [rnd_idx0, rnd_idx1, grounded_idx2]


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
        self.arc = 2*math.pi/settings.number_of_sensors #rad
        self.heading_coeff = 1
        self.safety_coeff = 4
        self.safety_dist = 1.5
        self.previous_obs = [3]*(settings.number_of_sensors+6)
        self.bug = tangent_bug()




    def predict(self, obs, goal):
        '''
        observation is in the form [angle, d_goal, vel_norm, vel_angle, y_pos, x_pos, d1, d2, ..., dn] where d1 starts at 180 deg and goes ccw, velocities are in drone's body frame ref
        actions are distributed as following:
        0-15: small circle
        16-31: medium small circle
        32-47: medium big circle
        48-63: big circle
        '''
        obs = obs[0][0] #flattening the list

        
        #read goal from observation (when not using tangent bug)
        #goal_angle = obs[0]*math.pi #rad
        #global_goal_distance = obs[1]

        #read goal coordinates from tangent bug
        x_goal = goal[0]
        y_goal = goal[1]
        global_goal_distance = np.sqrt(x_goal**2 + y_goal**2)
        #print(f"received goal (relative): {[x_goal,y_goal]}")

        
        vel_angle = obs[3]
        vel_norm = obs[2]
        x_pos = obs[5]
        y_pos = obs[4]
        predicted_delay = settings.delay*5 #accouting for predicted latency, and simulation time vs real time
        x_offset = predicted_delay*vel_norm*math.cos(vel_angle)*1.25
        y_offset = predicted_delay*vel_norm*math.sin(vel_angle)*1.25

        sensors = obs[6:settings.number_of_sensors+6] 
        #angles = obs[settings.number_of_sensors+6:]
        angles =  np.arange(-math.pi,math.pi,self.arc)
        #print(f"sensors: {np.round(sensors,1)}")

        # ---------------- random and greedy baselines -----------------------------
        if(msgs.algo == "GOFAI"):
            #chooses k closest sensors
            k_sensors = 12
            #chosen_idx = np.argpartition(sensors, k_sensors)[:k_sensors]
            #sensor_output = np.ones(settings.number_of_sensors)*100
            #for idx in chosen_idx:
            #    sensor_output[idx] = sensors[idx]
            #sensors = sensor_output
            #randomly chooses a subset of sensors to process (imitating RL agent)
            #n_sensors = 6
            #chosens = random.sample(range(len(sensors)),k=(settings.number_of_sensors-n_sensors))
            ##print(chosens)
            #for idx in chosens:
            #    sensors[idx] = 100
        #print(f"sensors dwa: {np.round(sensors,1)}")
        # -----------------------------------------------------------------


        objects =[]
        orientations = []
        #create objects list to evaluate obstacles positions, and replace missing values with old observations
        #values over 99 are the sensors that are "removed" by the RL agent
        for i, sensor in enumerate(sensors):
            if sensor < 99:
                if sensor >= 66:
                    sensors[i] = self.previous_obs[i]
                objects.append(sensors[i])
                orientations.append(angles[i])

        x_objects = []
        y_objects = []
        for object,angle in zip(objects,orientations):
            x_objects.append(object*math.cos(angle+self.arc/2) - x_offset)
            y_objects.append(object*math.sin(angle+self.arc/2) - y_offset)
        x_objects = np.array(x_objects)
        y_objects = np.array(y_objects)
            
        
        #print(f"dwa objects: {np.round(objects,1)}")
        #print(orientations)
        #print(len(objects))
        if len(objects) == 0: #if there is no obstacles, go straight to the goal at max speed
            thetas =  np.linspace(-math.pi, math.pi, settings.action_discretization+1)
            goal_angle = math.atan2(y_goal,x_goal)
            direction = bisect(thetas, goal_angle)
            if (thetas[direction]-goal_angle > goal_angle - thetas[direction-1]): #find which discretized value is closest
                direction -= 1
            action = (16 - direction%settings.action_discretization + round(0.75*settings.action_discretization))%settings.action_discretization  #transform the airsim action space (starts at 90 deg and goes cw)
            return action + 48
        
        #sensors = np.concatenate((sensors,sensors)) #this way we can more easily slice the angles we want
        #angles = np.concatenate((angles,angles))
        bestBenefit = -1000
        action = 0
        angle_increment = 2*math.pi/settings.action_discretization
        for i in range(settings.action_discretization*4): #4 velocities time 16 directions
            theta = math.pi/2 - angle_increment*(i%settings.action_discretization)  #in the action space, the circle starts at 90 deg and goes cw (drone body frame reference)

            #computing new distance to goal
            travel_speed = min(2, settings.base_speed*3**(i//settings.action_discretization)) #travelling speed can be 0.1, 0.3, 0.9, or 2 m/s 
            x_dest = travel_speed*math.cos(theta)*0.4*(settings.mv_fw_dur+predicted_delay*0.25) + vel_norm*math.cos(vel_angle)*(0.75+predicted_delay*1.25) # correcting for current speed since change in speed isn't instantaneous
            y_dest = travel_speed*math.sin(theta)*0.4*(settings.mv_fw_dur+predicted_delay*0.25) + vel_norm*math.sin(vel_angle)*(0.75+predicted_delay*1.25)

            new_dist = np.sqrt((x_goal-x_dest)**2+(y_goal-y_dest)**2)
            #computing the closest obstacle to the trajectory
            minDist = self.safety_dist
            if (len(objects) > 0):
                dist = self.shortest_distance_on_trajectory(x_objects,y_objects,x_dest,y_dest)
                if dist < minDist:
                    minDist = dist

            #computing the benefit
            benefit = self.heading_coeff*(global_goal_distance-new_dist) - self.safety_coeff*(self.safety_dist - minDist)
            #print(f"heading term: {global_goal_distance-new_dist}")
            #print(f"safety term: {self.safety_dist - minDist}")
            if benefit > bestBenefit:
                bestBenefit = benefit
                mindistAction = minDist
                headingTerm = self.heading_coeff*(global_goal_distance-new_dist)
                safetyTerm = self.safety_coeff*(self.safety_dist - minDist)
                action =i
                direction = theta


        self.previous_obs = sensors
        #print(f"min predicted distance in chosen dwa action: {mindistAction}")
        #print(f"heading term: {headingTerm} safety term: {safetyTerm}")
        #print(f"full loop action: {action}")

        ### -----------printing info on the chosen action-------------------------------------------------------------
        travel_speed = min(2, settings.base_speed*3**(action//settings.action_discretization)) #travelling speed can be 0.5, 1, 2, or 4 
        x_dest = travel_speed*math.cos(direction)*0.4*(settings.mv_fw_dur+predicted_delay*0.5) + vel_norm*math.cos(vel_angle) * (0.75+predicted_delay)  + x_pos # correcting for current speed since change in speed isn't instantaneous
        y_dest = travel_speed*math.sin(direction)*0.4*(settings.mv_fw_dur+predicted_delay*0.5)  + vel_norm*math.sin(vel_angle) * (0.75+predicted_delay) + y_pos
        print(f"dwa desired angle: {np.round(direction*180/math.pi,1)}")
        print(f"dwa action: {action}")
        #print(f"current speed: {[np.round(vel_norm,1), np.round(vel_angle*180/np.pi,1)]}")
        #print(f"min distance in chosen trajectory: {np.round(minDist,5)}")
        #print(f"objects: {np.round(objects,1)}")
        #print(f"orientations: {np.round(orientations,2)}")
        #print(f"sensors: {np.round(sensors,1)}")
        #print(f"goal_distance: {global_goal_distance} angle: {goal_angle*180/math.pi}")
        #print(f"destination: {[np.round(y_dest,1), np.round(x_dest,1)]}")
        #print(f"destination: {np.round(now,2)}")
        #print(f"min distance in chosen trajectory: {minDist}")
        #print(f"goal speed: {travel_speed}")
        #print(f"received speed: {np.round(np.sqrt(x_vel**2 + y_vel**2),2)}")
        #print(f"received pos: {[np.round(y_pos,2), np.round(x_pos,2)]}")
        #print(f"corrected pos: {[np.round(y_pos+y_offset,2), np.round(x_pos+x_offset,2)]}")
        #print(f"predicted destination: {[np.round(y_dest,2), np.round(x_dest,2)]}")
        #print(f"new_dist: {new_dist}")
        #---------------------------------------------

        
        return action

    def shortest_distance_on_trajectory(self, X, Y, x2, y2):
        """
        finds the closest point from the given points (X,Y) to the (x2,y2) line segment (from the origin). outputs the closest distance to that segment
        """
        norm = x2*x2 + y2*y2
        if norm == 0:
            return np.sqrt(np.min(X**2+Y**2))

        dotProducts = X*x2 + Y*y2

        params = np.ones(X.size)*-1
        params = np.clip(dotProducts/norm,0,1)

        xx = params*x2
        yy = params*y2

        dx = X - xx
        dy = Y - yy

        norms = dx**2 + dy**2
        minDist = np.sqrt(norms.min())

        return minDist



class APF():

    def __init__(self):
        self.arc = 2*math.pi/settings.number_of_sensors #rad
        self.previous_obs = [3]*(settings.number_of_sensors+6)
        self.attractive_coeff = 0.25
        self.repulsive_coeff = 3
        self.safety_dist = 5

    def attractive_force(self, position, goal):
        "Computes the attractive force to the goal. takes goal and robot positions as [x,y] arrays and returns [x,y] force"
        return self.attractive_coeff * (goal - position)

    def repulsive_force(self, position, obstacles):
        "Computes the repulsive forces to the goal. takes relative obstacles positions as a list of [x,y] arrays and returns [x,y] force"
        total_force = np.zeros_like(position)
        distances = np.linalg.norm(position - obstacles, axis=0)
        repulsion = self.repulsive_coeff * (1 / distances - 1 / self.safety_dist) * ((1 / (distances)) ** 2) * ((position - obstacles) / distances)
        total_force = np.sum(repulsion,axis=0)
        #print(f" for obstacle at {obstacle}, repulsive force is {repulsion}")
        return total_force

    def artificial_potential_field(self, position, goal, obstacles):
        attractive = self.attractive_force(position, goal)
        repulsive = self.repulsive_force(position, obstacles)
        total_force = attractive + repulsive
        return total_force

    
    def predict(self, obs, goal):
        '''
        observation is in the form [angle, d_goal, vel_norm, vel_angle, y_pos, x_pos, d1, d2, ..., dn] where d1 starts at 180 deg and goes ccw, velocities are in drone's body frame ref
        actions are distributed as following:
        0-15: small circle
        16-31: medium small circle
        32-47: medium big circle
        48-63: big circle
        '''
        obs = obs[0][0] #flattening the list

        
        #read goal from observation (when not using tangent bug)
        #goal_angle = obs[0]*math.pi #rad
        #global_goal_distance = obs[1]

        #read goal coordinates from tangent bug
        x_goal = goal[0]
        y_goal = goal[1]
        global_goal_distance = np.sqrt(x_goal**2 + y_goal**2)
        #print(f"received goal (relative): {[x_goal,y_goal]}")

        
        vel_angle = obs[3]
        vel_norm = obs[2]
        x_pos = obs[5]
        y_pos = obs[4]
        predicted_delay = settings.delay*5 #accouting for predicted latency, and simulation time vs real time
        x_offset = predicted_delay*vel_norm*math.cos(vel_angle)*1.25
        y_offset = predicted_delay*vel_norm*math.sin(vel_angle)*1.25

        sensors = obs[6:settings.number_of_sensors+6] 
        #angles = obs[settings.number_of_sensors+6:]
        angles =  np.arange(-math.pi,math.pi,self.arc)
        #print(f"sensors: {np.round(sensors,1)}")

        # ---------------- random and greedy baselines -----------------------------
        if(msgs.algo == "GOFAI"):
            #chooses k closest sensors
            k_sensors = 12
            #chosen_idx = np.argpartition(sensors, k_sensors)[:k_sensors]
            #sensor_output = np.ones(settings.number_of_sensors)*100
            #for idx in chosen_idx:
            #    sensor_output[idx] = sensors[idx]
            #sensors = sensor_output
            #randomly chooses a subset of sensors to process (imitating RL agent)
            #n_sensors = 6
            #chosens = random.sample(range(len(sensors)),k=(settings.number_of_sensors-n_sensors))
            ##print(chosens)
            #for idx in chosens:
            #    sensors[idx] = 100
        #print(f"sensors dwa: {np.round(sensors,1)}")
        # -----------------------------------------------------------------


        objects =[]
        orientations = []
        #create objects list to evaluate obstacles positions, and replace missing values with old observations
        #values over 99 are the sensors that are "removed" by the RL agent
        for i, sensor in enumerate(sensors):
            if sensor < 99:
                if sensor >= 66:
                    sensors[i] = self.previous_obs[i]
                object_x = math.cos(angles[i])*sensors[i]
                object_y = math.sin(angles[i])*sensors[i]
                objects.append(np.array([object_x, object_y]))
                orientations.append(angles[i])
            
        
        #print(f"dwa objects: {np.round(objects,1)}")
        #print(orientations)
        #print(len(objects))
        thetas =  np.linspace(-math.pi, math.pi, settings.action_discretization+1)
        if len(objects) == 0: #if there is no obstacles, go straight to the goal at max speed
            goal_angle = math.atan2(y_goal,x_goal)
            direction = bisect(thetas, goal_angle)%16
            if (thetas[direction]-goal_angle > goal_angle - thetas[direction-1]): #find which discretized value is closest
                direction -= 1
            action = (16 - direction%settings.action_discretization + round(0.75*settings.action_discretization))%settings.action_discretization  #transform the airsim action space (starts at 90 deg and goes cw)
            #print(f"Direction: {thetas[direction]*180/np.pi} at 2 m/s. action: {action+48}. no obstacles!")
            return action + 48

        force =  self.artificial_potential_field(np.zeros(2), np.array([x_goal, y_goal]), np.array(objects))
        #print(f"force [x,y]: {force}")
        force_angle = math.atan2(force[1],force[0])
        speed = np.linalg.norm(force)

        direction = bisect(thetas, force_angle)%16
        if (thetas[direction]-force_angle > force_angle - thetas[direction-1]): #find which discretized value is closest
            direction -= 1
        action = (16 - direction%settings.action_discretization + round(0.75*settings.action_discretization))%settings.action_discretization  #transform the airsim action space (starts at 90 deg and goes cw)
        #adjusting for force magnitude
        if action > 2:
            action += 3*settings.action_discretization
        elif action > 1:
            action += 2*settings.action_discretization
        elif action > 0.3:
            action += settings.action_discretization

        self.previous_obs = sensors
        #print(f"Direction: {thetas[direction]*180/np.pi%360} at {speed} m/s. action: {action}")

        return action