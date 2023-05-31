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

def intersection(agent, classical, random, greedy):
    """ 
    Finds the intersection of successful runs between the 4 methods and outputs only the values in that intersection, restricted with the episodes with similar starting position
    (previous episode was either as success or a fail for both methods)
    """

    agent_list = []
    classical_list  = []
    random_list = []
    greedy_list = []
    if agent[0] > 0 and classical[0] > 0 and random[0] > 0 and greedy[0] > 0:
        agent_list.append(agent[0])
        classical_list.append(classical[0])
        random_list.append(random[0])
        greedy_list.append(greedy[0])

    for i in range(1,len(agent)):
        if agent[i] > 0 and classical[i] > 0 and random[i] > 0 and greedy[i] > 0: #make sure all are successful
            #print(f"all are successful")
            if (agent[i-1] == classical[i-1] == random[i-1] == greedy[i-1]) or (agent[i-1] > 0 and classical[i-1] > 0 and random[i-1] > 0 and greedy[i-1] > 0): #make sure they start at similar positions
                agent_list.append(agent[i])
                classical_list.append(classical[i])
                random_list.append(random[i])
                greedy_list.append(greedy[i])

    return agent_list, classical_list, random_list, greedy_list

if __name__ == '__main__':
    if len(sys.argv) == 3:
        agent_traveled_distances, agent_ideal_distances, agent_mission_times = parse(sys.argv[1])
        baseline_traveled_distances, baseline_ideal_distances, baseline_mission_times = parse(sys.argv[2])
    else:
        agent_traveled_distances, agent_ideal_distances, agent_mission_times = parse("C:/Users/charl/workspace/airlearning/airlearning-rl/data/A2C-B/metrics.txt")
        classical_traveled_distances, classical_ideal_distances, classical_mission_times = parse("C:/Users/charl/workspace/airlearning/airlearning-rl/data/GOFAI/metrics_classical.txt")
        random_traveled_distances, random_ideal_distances, random_mission_times = parse("C:/Users/charl/workspace/airlearning/airlearning-rl/data/GOFAI/metrics_random.txt")
        greedy_traveled_distances, greedy_ideal_distances, greedy_mission_times = parse("C:/Users/charl/workspace/airlearning/airlearning-rl/data/GOFAI/metrics_greedy.txt")


    agent_traveled_intersection, classical_traveled_intersection, random_traveled_intersection, greedy_traveled_intersection = intersection(agent_traveled_distances, classical_traveled_distances, random_traveled_distances, greedy_traveled_distances)
    agent_ideal_intersection, classical_ideal_intersection, random_ideal_intersection, greedy_ideal_intersection = intersection(agent_ideal_distances, classical_ideal_distances, random_ideal_distances, greedy_ideal_distances)
    agent_time_intersection, classical_time_intersection, random_time_intersection, greedy_time_intersection = intersection(agent_mission_times, classical_mission_times, random_mission_times, greedy_mission_times)


    episodes = len(agent_traveled_intersection)
    agent_ratios = [traveled/ideal for (traveled,ideal) in zip(agent_traveled_intersection, agent_ideal_intersection)]
    classical_ratios = [traveled/ideal for (traveled,ideal) in zip(classical_traveled_intersection, classical_ideal_intersection)]
    random_ratios = [traveled/ideal for (traveled,ideal) in zip(random_traveled_intersection, random_ideal_intersection)]
    greedy_ratios = [traveled/ideal for (traveled,ideal) in zip(greedy_traveled_intersection, greedy_ideal_intersection)]
    plt.figure()
    plt.plot(range(episodes), agent_ratios, range(episodes), classical_ratios, range(episodes), random_ratios, range(episodes), greedy_ratios)
    plt.legend(['agent','classical','random','greedy'])
    plt.xlabel('distance travelled')
    plt.ylabel('ratio')
    plt.title('distances traveled / birdview distances for the episodes in the successful intersection')

    plt.figure()
    plt.plot(range(episodes), agent_traveled_intersection, range(episodes), classical_traveled_intersection, range(episodes), random_traveled_intersection, range(episodes), greedy_traveled_intersection)
    plt.legend(['agent','classical','random','greedy'])
    plt.xlabel('episodes')
    plt.ylabel('distance')
    plt.title('distances traveled for the episodes in the successful intersection')

    #plt.figure()
    #plt.plot(range(episodes), agent_ideal_intersection, range(episodes), baseline_ideal_intersection)
    #plt.legend(['agent','baseline'])
    #plt.xlabel('episodes')
    #plt.ylabel('distances')
    #plt.title('ideal distances traveled for the episodes in the successful intersection')

    plt.figure()
    plt.plot(range(episodes), agent_time_intersection, range(episodes), classical_time_intersection, range(episodes), random_time_intersection, range(episodes), greedy_time_intersection)
    plt.legend(['agent','classical','random','greedy'])
    plt.xlabel('episodes')
    plt.ylabel('time (s)')
    plt.title('mission time for the episodes in the successful intersection')

    print(f"agent total distance traveled in the intersection: {sum(agent_traveled_intersection)}")
    print(f"classical total distance traveled in the intersection: {sum(classical_traveled_intersection)}")
    print(f"random total distance traveled in the intersection: {sum(random_traveled_intersection)}")
    print(f"greedy total distance traveled in the intersection: {sum(greedy_traveled_intersection)}")

    print(f"agent total mission time in the intersection: {sum(agent_time_intersection)}")
    print(f"classical total mission time in the intersection: {sum(classical_time_intersection)}")
    print(f"random total mission time in the intersection: {sum(random_time_intersection)}")
    print(f"greedy total mission time in the intersection: {sum(greedy_time_intersection)}")

    plt.show()