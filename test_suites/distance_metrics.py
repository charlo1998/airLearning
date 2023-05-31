import os
import matplotlib.pyplot as plt
import numpy as np
import ast
import sys

def parse(file):
    with open(file, 'r') as f:
        traveled_distances = ""
        ideal_distances = ""
        mission_times = ""

        line = f.readline()
        if 'traveled_distances' in line: #found the first list
            traveled_distances = line.replace("traveled_distances:", "")
            print("finished reading traveled_distances")
            line = f.readline()
            ideal_distances = line.replace("ideal_distances:", "")
            print("finished reading ideal_distances")
            line = f.readline()
            mission_times = line.replace("mission_times:", "")
            print("finished reading mission_times")


        #removing brackets and spaces
        print("processing data")
        traveled_distances = traveled_distances.strip("[ ]\n")
        ideal_distances = ideal_distances.strip("[ ]\n")
        mission_times = mission_times.strip("[ ]\n")

        #converting into list of floats
        traveled_distances = [float(value) for value in traveled_distances.split(",")]
        ideal_distances = [float(value) for value in ideal_distances.split(",")]
        mission_times = [float(value) for value in mission_times.split(",")]


    return traveled_distances, ideal_distances, mission_times

def intersection(agent, baseline):
    """ 
    Finds the intersection of successful runs between the two methods and outputs only the values in that intersection, restricted with the episodes with similar starting position
    (previous episode was either as success or a fail for both methods)
    """

    agent_list = []
    baseline_list  = []
    if agent[0] > 0 and baseline[0] > 0:
        agent_list.append(agent[0])
        baseline_list.append(baseline[0])

    for i in range(1,len(agent)):
        if agent[i] > 0 and baseline[i] > 0: #make sure both are successful
            if (agent[i-1] == baseline[i-1]) or (agent[i-1] > 0 and baseline[i-1] > 0): #make sure they start at similar positions
                agent_list.append(agent[i])
                baseline_list.append(baseline[i])

    return agent_list, baseline_list

if __name__ == '__main__':
    if len(sys.argv) == 3:
        agent_traveled_distances, agent_ideal_distances, agent_mission_times = parse(sys.argv[1])
        baseline_traveled_distances, baseline_ideal_distances, baseline_mission_times = parse(sys.argv[2])
    else:
        agent_traveled_distances, agent_ideal_distances, agent_mission_times = parse("C:/Users/charl/workspace/airlearning/airlearning-rl/data/A2C-B/metrics.txt")
        baseline_traveled_distances, baseline_ideal_distances, baseline_mission_times = parse("C:/Users/charl/workspace/airlearning/airlearning-rl/data/GOFAI/metrics.txt")

    agent_traveled_intersection, baseline_traveled_intersection = intersection(agent_traveled_distances, baseline_traveled_distances)
    agent_ideal_intersection, baseline_ideal_intersection = intersection(agent_ideal_distances, baseline_ideal_distances)
    agent_time_intersection, baseline_time_intersection = intersection(agent_mission_times, baseline_mission_times)


    episodes = len(agent_traveled_intersection)
    agent_ratios = [traveled/ideal for (traveled,ideal) in zip(agent_traveled_intersection, agent_ideal_intersection)]
    baseline_ratios = [traveled/ideal for (traveled,ideal) in zip(baseline_traveled_intersection, baseline_ideal_intersection)]
    plt.figure()
    plt.plot(range(episodes), agent_ratios, range(episodes), baseline_ratios)
    plt.legend(['agent','baseline'])
    plt.xlabel('distance travelled')
    plt.ylabel('ratio')
    plt.title('distances traveled / birdview distances for the episodes in the successful intersection')

    plt.figure()
    plt.plot(range(episodes), agent_traveled_intersection, range(episodes), baseline_traveled_intersection)
    plt.legend(['agent','baseline'])
    plt.xlabel('episodes')
    plt.ylabel('distance')
    plt.title('distances traveled for the episodes in the successful intersection')

    plt.figure()
    plt.plot(range(episodes), agent_ideal_intersection, range(episodes), classical_ideal_intersection, range(episodes), random_ideal_intersection, range(episodes), greedy_ideal_intersection)
    plt.legend(['agent','classical','random','greedy'])
    plt.xlabel('episodes')
    plt.ylabel('distances')
    plt.title('ideal distances traveled for the episodes in the successful intersection')

    plt.figure()
    plt.plot(range(episodes), agent_time_intersection, range(episodes), baseline_time_intersection)
    plt.legend(['agent','baseline'])
    plt.xlabel('episodes')
    plt.ylabel('time (s)')
    plt.title('mission time for the episodes in the successful intersection')

    print(f"agent total distance traveled in the intersection: {sum(agent_traveled_intersection)}")
    print(f"baseline total distance traveled in the intersection: {sum(baseline_traveled_intersection)}")
    print(f"agent total mission time in the intersection: {sum(agent_time_intersection)}")
    print(f"baseline total mission time in the intersection: {sum(baseline_time_intersection)}")

    plt.show()