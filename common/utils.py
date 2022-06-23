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

def parse_data(file_name):
    with open(file_name  , 'a+') as f:
        f.seek(0,0)
        lines = f.readlines()
        if lines[-1] != '}':
            print("added } at log eof!")
            f.write("\n}")
        else:
            print("already well formatted")

    if (file_name == ''):
        file_hndl = open(os.path.join(settings.proj_root_path, "data", msgs.algo, msgs.mode + "_episodal_log.txt"), "r")
    else:
        file_hndl = open(file_name, "r")
    #print(f"parsing: {file_name}")
    data = json.load(file_hndl)
    data_clusterd_based_on_key = {}

    data.pop('buffer size', None) #removing settings data from the training data to be parsed
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


def plot_data(file, data_to_inquire, mode="separate"):
    # santize_data(file)
    #make a list of dictionnaries, loop using setting.runs_to_do and average the values
    dataList = []
    for i in range(settings.runs_to_do):
        dataList.append(parse_data(file.replace("log", "log" + str(i))))
    #print(dataList)
    data = dataList #to do, average the values instead of plotting them all. warning: the runs have different length of episodes!
    for el in data_to_inquire:
        #print(el)
        for i in range(settings.runs_to_do):
            #print(data[i][el[1]])
            plt.plot(data[i][el[0]], data[i][el[1]])
            assert (el[0] in data[i].keys())
        plt.xlabel(el[0])
        plt.ylabel(el[1])
        
        plt.legend()
        plt.draw()
        plt.figure()
    if (mode == "separate"):
        plt.show()
    # plt.draw()
    # plt.pause(.001)


def generate_csv(file):
    data = parse_data(file)
    data_frame = pd.DataFrame(data)
    data_frame.to_csv(file.replace("txt", "csv"), index=False)


def append_log_file(episodeN, log_mode="verbose"):
    with open(os.path.join(settings.proj_root_path, "data", msgs.algo, msgs.mode + "_episodal_log" + log_mode + str(settings.i_run) + ".txt"),
              "a+") as f:
        if (episodeN == 0): #starting a new training task, add settings 
            f.write('{\n')
            f.write('"buffer size":' + str(settings.buffer_size) + ',\n')
            f.write('"loaded from checkpoint":' + str(settings.use_checkpoint).lower() + ',\n')
            file_hndl = open("C:/Users/charl/Documents/AirSim/settings.json", "r")
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

    if (settings.end_randomization_mode == "inclusive"):
        idx0_low_pos_bndry = 0
        idx1_low_pos_bndry = 0
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

    return [rnd_idx0, rnd_idx1, grounded_idx2]


# return [rnd_idx0, rnd_idx1, rnd_idx2]


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
