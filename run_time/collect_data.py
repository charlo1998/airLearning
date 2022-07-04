import os

os.sys.path.insert(0, os.path.abspath('../settings_folder'))

import settings

import dqn_airsim
import ddpg_airsim
import dqn_baselines
import ppo_airsim
#import sac_airsim
from game_handler_class import *
import msgs
import file_handling
from utils import *


def runTask(task):
    # decide on the algorithm
    # DQN-B is the stable-baselines version of DQN
    # DQN is the Keras-RL version of DQN
    game_handler = GameHandler()

    if ("algo" in task.keys()):
        if (task["algo"] in ["DDPG", "DQN", "PPO", "SAC", "DQN-B"]):
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
            elif (task["algo"] == "SAC"):
                train_class = sac_airsim #not available?
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
        elif task["algo"] == "DQN-B":
            train_class.train(train_obj, env)

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


def main():
    taskList = []
    #model_weights_list_to_test = ["C:/Users/charl/workspace/airlearning/airlearning-rl/data/DQN-B/model.pkl"] #baselines
    model_weights_list_to_test = ["C:/Users/charl/workspace/airlearning/airlearning-rl/run_time/saved_model/dqn_weights_run0.hf5"] #keras rl

    algo = "DQN"
    task_type = "train"

    task1 = {"task_type": "start_game"}
    task2 = {"algo": algo, "task_type": task_type, "difficulty_level": "default", "env_name": "AirSimEnv-v42",
             "weights": model_weights_list_to_test}
    task3 = {"task_type": "kill_game"}
    task4 = {"algo": algo, "task_type": "generate_csv", "data_file": task_type + "_episodal_log.txt"}
    task5 = {"algo": algo, "task_type": "plot_data", "data_file": task_type + "_episodal_log.txt", "data_to_plot": [["episodeN", "success_ratio_within_window"], ["total_step_count_for_experiment", "total_reward"]], "plot_data_mode": "separate"}
    
    taskList.append(task1)

    if task_type == "train":
        for i in range(settings.runs_to_do):
            taskList.append(task2) #train
    else:
        taskList.append(task2) # don't do multiple runs for test

    taskList.append(task3) #close airlearning
    #taskList.append(task4) #generate_csv
    taskList.append(task5) #plot

    for task_el in taskList:
        runTask(task_el)


if __name__ == "__main__":
    main()
