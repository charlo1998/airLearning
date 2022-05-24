import os

os.sys.path.insert(0, os.path.abspath('../settings_folder'))

import settings
import ddpg_airsim
import dqn_airsim
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
    if ("algo" in task.keys()):
        if (task["algo"] in ["DDPG", "DQN", "PPO", "SAC", "DQN-B"]):
            if (task["algo"] == "DDPG"):
                msgs.algo = "DDPG"
                train_class = ddpg_airsim
            elif (task["algo"] == "PPO"):
                msgs.algo = "PPO"
                train_class = ppo_airsim
            elif (task["algo"] == "DQN"):
                train_class = dqn_airsim
                msgs.algo = "DQN"
            elif (task["algo"] == "DQN-B"):
                train_class = dqn_baselines
                msgs.algo = "DQN-B"
            elif (task["algo"] == "SAC"):
                train_class = sac_airsim
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
        if (task["algo"] == "DQN"):
            train_class.train(train_obj, env)

        if(task["algo"] == "DQN-B"):
            train_class.train(train_obj, env)

        if (task["algo"] == "PPO"):
            train_class.train(train_obj, env)

        if (task["algo"] == "SAC"):
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
        game_handler = GameHandler()
        game_handler.start_game_in_editor()

    if (task["task_type"] == "restart_game"):
        game_handler = GameHandler()
        game_handler.restart_game()

    if task["task_type"] == "generate_csv":
        msgs.algo = task["algo"]
        csv_file = os.path.join(settings.proj_root_path, "data", msgs.algo, task["data_file"])
        generate_csv(csv_file)

    if task["task_type"] == "plot_data":
        data_file = os.path.join(settings.proj_root_path, "data", task["algo"], task["data_file"])
        plot_data(data_file, task["data_to_plot"], task["plot_data_mode"])


def main():
    taskList = []
    model_weights_list_to_test = ["C:/Users/charl/workspace/airlearning/airlearning-rl/data/DQN-B/model.pkl"]

    algo = "DQN"

    task1 = {"task_type": "start_game"}
    task2 = {"algo": algo, "task_type": "train", "difficulty_level": "default", "env_name": "AirSimEnv-v42",
             "weights": model_weights_list_to_test}
    task3 = {"algo": algo, "task_type": "test", "difficulty_level": "default", "env_name": "AirSimEnv-v42",
             "weights": model_weights_list_to_test}
    task4 = {"algo": algo, "task_type": "generate_csv", "data_file": "train_episodal_log.txt"}
    task5 = {"algo": algo, "task_type": "plot_data", "data_file": "train_episodal_log.txt", "data_to_plot": [["episodeN", "success_ratio_within_window"]], "plot_data_mode": "separate"}
    
    taskList.append(task1)
    taskList.append(task2)
    #taskList.append(task3)
    taskList.append(task4)
    taskList.append(task5)

    for task_el in taskList:
        runTask(task_el)


if __name__ == "__main__":
    main()
