import os
import matplotlib.pyplot as plt
import numpy as np
import ast
import sys


def Parse(file):

    with open(file, 'r') as f:
        loop_rate_list = ""
        take_action_list = ""
        clct_state_list = ""

        line = f.readline()
        if 'loop_rate_list' in line: #found the first list
            line = line.replace("loop_rate_list:", "")
            while 'take_action_list' not in line: #continue reading until start of next list
                loop_rate_list = loop_rate_list + line
                line = f.readline()
            line = line.replace("take_action_list:", "")
            while 'clct_state_list' not in line: #continue reading until start of next list
                take_action_list = take_action_list + line
                line = f.readline()
            line = line.replace("clct_state_list:", "")
            clct_state_list = line + f.read() #read until end of file

        #loop_rate_out = ast.literal_eval(loop_rate_list)
        #clct_state_out = ast.literal_eval(clct_state_list)
        #take_action_out = ast.literal_eval(take_action_list)

        #removing brackets and spaces
        loop_rate_out = loop_rate_list.strip("[ ]\n")
        clct_state_out = clct_state_list.strip("[ ]\n")
        take_action_out = take_action_list.strip("[ ]\n")

        #converting into list of floats
        loop_rate_out = [float(value) for value in loop_rate_out.split(",")]
        clct_state_out = [float(value) for value in clct_state_out.split(",")]
        take_action_out = [float(value) for value in take_action_out.split(",")]


    return take_action_out, clct_state_out, loop_rate_out

def SmoothData(take_action_list, clct_state_list, loop_rate, nb_to_avg = 200):
    
    avg_actions = np.zeros(len(take_action_list[nb_to_avg:]))
    avg_collect = np.zeros(len(clct_state_list[nb_to_avg:]))
    avg_loop = np.zeros(len(loop_rate[nb_to_avg:]))


    for i in range(len(take_action_list[nb_to_avg:])):
        if take_action_list[i+nb_to_avg] >= avg_actions[i-1]*3:
            take_action_list[i+nb_to_avg] = take_action_list[i]
        avg_actions[i] = np.sum(take_action_list[i:i+nb_to_avg])/nb_to_avg 
    for i in range(len(clct_state_list[nb_to_avg:])):
        if clct_state_list[i+nb_to_avg] >= avg_collect[i-1]*3:
            clct_state_list[i+nb_to_avg] = clct_state_list[i]
        avg_collect[i] = np.sum(clct_state_list[i:i+nb_to_avg])/nb_to_avg
    for i in range(len(loop_rate[nb_to_avg:])):
        if loop_rate[i+nb_to_avg] >= avg_loop[i-1]*3:
            loop_rate[i+nb_to_avg] = loop_rate[i]
        avg_loop[i] = np.sum(loop_rate[i:i+nb_to_avg])/nb_to_avg

    return avg_actions*1000.0, avg_collect*1000.0, avg_loop*1000.0 #convert to milliseconds


def plot(data="env_log.txt"):
    take_action_list, clct_state_list, loop_rate = Parse(data)
    avg_actions, avg_collect, avg_loop = SmoothData(take_action_list, clct_state_list, loop_rate, 50)
    print(f"total time elapsed iterating: {sum(loop_rate)}")

    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    ax1.plot(range(len(avg_actions)), avg_actions, label='take_action')
    ax1.plot(range(len(avg_collect)), avg_collect, label='collect_state')
    ax1.plot(range(len(avg_loop)), avg_loop, label='step_loop')
    #ax1.set_ylim([0,1.5])
    ax1.set_ylabel("time (ms)")
    ax1.set_xlabel("simulation timesteps")
    ax1.set_title("linear scale")
    ax1.legend()

    ax2.plot(range(len(avg_loop)), np.sqrt(avg_loop), label='step_loop')
    ax2.set_title("quadratic scale")
    ax2.set_ylabel("sqrt(time) (ms)")
    ax2.set_xlabel("simulation timesteps")
    #ax2.set_ylim([0.2,1.2])

    ax3.plot(range(len(avg_loop)), np.log(avg_loop), label='step_loop')
    ax3.set_ylabel("log(time) (ms)")
    ax3.set_title("exponential scale")
    ax3.set_xlabel("simulation timesteps")
    #ax3.set_ylim([-2.6,0.6])
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) == 2:
        plot(sys.argv[1])
    else:
        plot()