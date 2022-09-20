import os

os.sys.path.insert(0, os.path.abspath('../settings_folder'))

import settings
import msgs
import dqn_airsim
import ddpg_airsim
import dqn_baselines
import ppo_airsim
from algorithms.discrete import a2c_baselines
#import sac_airsim
from game_handler_class import *

import file_handling
from utils import *


def runTask(task):
    # decide on the algorithm
    # DQN-B is the stable-baselines version of DQN
    # DQN is the Keras-RL version of DQN
    game_handler = GameHandler()

    if ("algo" in task.keys()):
        if (task["algo"] in ["DDPG", "DQN", "PPO", "A2C-B", "DQN-B"]):
            if (task["algo"] == "DDPG"):
                msgs.algo = "DDPG"
                train_class = ddpg_airsim #ddpg not working? issue when creating the actor network
            elif (task["algo"] == "PPO"):
                msgs.algo = "PPO" #reeeeeeally slow?
                train_class = ppo_airsim
            elif (task["algo"] == "DQN"):
                train_class = dqn_airsim
                msgs.algo = "DQN"
            elif (task["algo"] == "DQN-B"):
                train_class = dqn_baselines
                msgs.algo = "DQN-B"
            elif (task["algo"] == "A2C-B"):
                train_class = a2c_baselines
                msgs.algo = "A2C-B"
        else:
            print("this algorithm is not supported")
            exit(0)

    if (task["task_type"] == "backup"):
        backup_obj = file_handling.Backup()
        backup_obj.get_backup()

    if (task["task_type"] == "train"):
        print("setting up training")
        train_obj, env = train_class.setup(env_name=task["env_name"], \
                                           difficulty_level=task["difficulty_level"])
        print("starting training")
        if task["algo"] == "DQN":
            train_class.train(train_obj, env, train_checkpoint = settings.use_checkpoint)
        elif task["algo"] == "DQN-B" or task["algo"] == "A2C-B":
            train_class.train(train_obj, env, checkpoint = task["checkpoint"]) #only will use the checkpoint if settings.checkpoint = True

    if (task["task_type"] == "test"):

        if (len(task["weights"]) == 0):
            task["weights"] = file_handling.find_all_weight_files(msgs.algo, settings.proj_root_path)

        for weights in task["weights"]:
            utils.reset_msg_logs()
            train_obj, env = train_class.setup(env_name=task["env_name"], \
                                               difficulty_level=task["difficulty_level"])
            train_class.test(train_obj, env, weights)

    if (task["task_type"] == "start_game"):
        game_handler.start_game_in_editor()

    if (task["task_type"] == "restart_game"):
        game_handler.restart_game()

    if (task["task_type"] == "kill_game"):
        game_handler.kill_game_in_editor()

    if task["task_type"] == "generate_csv":
        msgs.algo = task["algo"]
        csv_file = os.path.join(settings.proj_root_path, "data", msgs.algo, task["data_file"])
        generate_csv(csv_file)

    if task["task_type"] == "plot_data":
        data_file = os.path.join(settings.proj_root_path, "data", task["algo"], task["data_file"])
        plot_data(data_file, task["data_to_plot"], task["plot_data_mode"])

    if task["task_type"] == "plot_trajectories":
        data_file = os.path.join(settings.proj_root_path, "data", task["algo"], task["data_file"])
        plot_trajectories(data_file)

def main():
    taskList = []

    #put weights to test in a list as we can test multiple in one task
    model_weights_list_to_test = ["C:/Users/charl/workspace/airlearning/airlearning-rl/data/A2C-B/model"] #baselines
    
    model_to_checkpoint = "C:/Users/charl/workspace/airlearning/airlearning-rl/data/A2C-B/model"

    algo = "A2C-B"
    task_type = "test"

    task1 = {"task_type": "start_game"}
    task2 = {"algo": algo, "task_type": task_type, "difficulty_level": settings.difficulty, "env_name": "AirSimEnv-v42",
             "weights": model_weights_list_to_test, "checkpoint": model_to_checkpoint}
    task3 = {"task_type": "kill_game"}
    task4 = {"algo": algo, "task_type": "generate_csv", "data_file": task_type + "_episodal_log.txt"}
    task5 = {"algo": algo, "task_type": "plot_data", "data_file": task_type + "_episodal_log.txt", "data_to_plot": [["episodeN", "success_ratio_within_window"], ["total_step_count_for_experiment", "total_reward"], ["episodeN", "stepN"]], "plot_data_mode": "separate"}
    task6 = {"algo": algo, "task_type": "plot_trajectories", "data_file": task_type + "_episodal_logverbose0.txt"}




    taskList.append(task1) #start gane
    if task_type == "train":
        for i in range(settings.runs_to_do):
            taskList.append(task2) #train
    else:
        taskList.append(task2) # don't do multiple runs for test
        taskList.append(task6) #plot trajectories

    taskList.append(task3) #close game
    
    taskList.append(task5) #plot
    #taskList.append(task4) #generate_csv

    for task_el in taskList:
        #print(f'executing task {task_el}')
        runTask(task_el)


if __name__ == "__main__":
    main()
