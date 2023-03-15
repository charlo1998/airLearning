import math
import numpy as np
import settings
import random
import msgs


class tangent_bug():
    '''
    implementation of a tangent bug algorithm for path planning with obstacle avoidance.
    '''

    def __init__(self):
        self.arc = 2*math.pi/settings.number_of_sensors #rad
        self.d_leave = 150
        self.d_min = 149
        self.following_boundary = False
        self.following_boundary_counter=0
        self.done =False
        self.min_dist = 150
        self.max_dist = 10
        self.previous_obs = [3]*(settings.number_of_sensors+4)
        self.foundPathCounter = 0
        self.tangent_direction = 1
        self.tangent_counter = 0



    def predict(self, obs):

        obs = obs[0][0] #flattening the list
        obs[6:] = 100**obs[6:] #reconverting from normalized to real values
        obs[1] = 100**obs[1]
        sensors = obs[6:]

        goal_angle = obs[0]*math.pi #rad
        goal_distance = obs[1]
        x_vel = obs[3]
        y_vel = obs[2]

        # ---------------- random baseline -----------------------------
        if(msgs.algo == "GOFAI"):
            #randomly chooses a subset of sensors to process (imitating RL agent)
            n_sensors = 60
            chosens = random.sample(range(len(sensors)),k=(settings.number_of_sensors-n_sensors))
            #print(chosens)
            for idx in chosens:
                sensors[idx] = 100
        #print(f"sensors bug: {np.round(sensors,1)}")
        # -----------------------------------------------------------------

        angles =  np.arange(-math.pi,math.pi,self.arc)
        objects =[]
        orientations = []
        #create objects list to evaluate obstacles positions, and replace missing values with old observations.
        #values over 99 are the sensors that are "removed" by the RL agent
        #any distance greater than the treshold will be ceiled.
        for i, sensor in enumerate(sensors):
            if sensor < 99:
                if sensor >= 66:
                    sensors[i] = self.previous_obs[i]
                objects.append(min(sensors[i], self.max_dist))
                orientations.append(angles[i])

        segments = self.compute_discontinuities(objects)
        #print(objects)

        #print(f"sensors: {np.round(sensors,1)}")
        #print(f"distances: {np.round(objects,1)}")
        #print(f"segments: {segments}")
        print_angles = [x*180/math.pi for x in orientations]
        #print(f"angles: {np.round(print_angles,2)}")

        
        if self.done: #finished episode, reset distances
            self.d_leave = 150
            self.d_min = 149
            self.min_dist = 150
            self.following_boundary_counter = 0
            self.following_boundary = False
            self.foundPathCounter = 0
            self.tangent_direction = 1
            self.tangent_counter = 0

        #find direction that minimizes distance to goal
        foundPath = False
        min_heuristic = 150
        for i, object in enumerate(objects):
            heuristic = self.compute_heuristic(object, orientations[i], goal_distance, goal_angle)
            if heuristic <= min_heuristic:
                min_heuristic = heuristic
                direction = orientations[i]
                best_idx = i
        #print(f"previous heuristic: {self.min_dist}")
        if min_heuristic <= self.min_dist:
            self.min_dist = min_heuristic
            foundPath = True
            self.foundPathCounter = 0
        heuristic = self.compute_heuristic(objects[best_idx], orientations[best_idx], goal_distance, goal_angle)
        

        if foundPath == False and self.following_boundary == False:
            self.foundPathCounter += 1
            direction = math.pi/2 - goal_angle
            #print("heuristic increased, go straight into goal, counter increased for boundary following")
            goal = [goal_distance*math.cos(direction), goal_distance*math.sin(direction)]

        #if the heuristic didn't decrease after last couple actions, we need to enter into boundary following
        if self.foundPathCounter >= 8 and not self.following_boundary:
            print("entering boundary following")
            self.following_boundary = True
            self.following_boundary_counter=0
            self.tangent_counter = 0



        if(not self.following_boundary):
            
            #action = 12 - direction_idx + 32
            
            #print(f"direction: {np.round(direction*180/math.pi,2)}")
            if goal_distance > objects[best_idx]:
                goal = [objects[best_idx]*math.cos(direction), objects[best_idx]*math.sin(direction)]  #drone body frame ref
            else:
                goal = [goal_distance*math.cos(direction), goal_distance*math.sin(direction)]  #drone body frame ref
            #take moving action


        else:


            closest_obstacle_idx = np.argmin(objects)
            tangent = orientations[closest_obstacle_idx]+math.pi/2*self.tangent_direction
            
            goal = [settings.mv_fw_spd_1*math.cos(tangent), settings.mv_fw_spd_1*math.sin(tangent)]

            #print(f"closest obstacle is at angle {orientations[closest_obstacle_idx]*180/math.pi} and distance {objects[closest_obstacle_idx]}. Tangent:  {tangent*180/math.pi}")
            
            object_to_avoid = segments[closest_obstacle_idx]
            #print(f"avoiding segment no. : {object_to_avoid}")
            self.d_leave, direction, idx = self.compute_d_leave(objects, angles, goal_distance, goal_angle)
            self.d_min = self.compute_d_min(objects, angles, goal_distance, goal_angle, object_to_avoid, segments)
            #print(f"d_leave: {self.d_leave}  d_min: {self.d_min}")
            print(f"boundary folling counter: {self.following_boundary_counter}")
            print(f"tangent counter: {self.tangent_counter}")

            #check if goal reached or escape found, or far from any obstacles
            if (self.done or self.d_leave < self.d_min):
                self.following_boundary_counter += 1
                if goal_distance > objects[idx]:
                    goal = [objects[idx]*math.cos(direction), objects[idx]*math.sin(direction)]  #drone body frame ref
                else:
                    goal = [goal_distance*math.cos(direction), goal_distance*math.sin(direction)]  #drone body frame ref

                
                print(f"done: {self.done}")
                if self.following_boundary_counter > 3:
                    self.following_boundary = False
                    self.foundPathCounter = 0
                    print("switched back to normal path")

            else:
                self.following_boundary_counter = 0
                #if we've been following the boundary for a long time, try returning to normal state and switching tangent directions
                self.tangent_counter +=1
                if self.tangent_counter > 100:
                    self.tangent_counter = -100
                    self.tangent_direction = -1*self.tangent_direction
                    print("switching direction")
                    self.following_boundary = False
                    self.foundPathCounter = 0
                    self.d_leave = 150
                    self.d_min = 149
                    self.min_dist = 150
                    self.tangent_counter = 0
                    print("reset to normal behavior")
         
            


        self.previous_obs = sensors
        #print(f"goal distance: {goal_distance} angle: {goal_angle*180/math.pi}")
        #print(f"goal (conventional coordinates): {goal}")

        return goal




    def compute_d_leave(self, objects, angles, goal_dist, goal_angle):

        #if the bug sees a escape window between two obstacles, but it is too narrow for the dwa to enter it, it will fail to avoid the local minimum.
        #we want to mark it as "obstacle"
        temp_objects = objects.copy()
        for i, object in enumerate(objects):
            if i == 0:
                left = -1
                right = 1
            elif i == len(objects)-1:
                left = len(objects)-2
                right = 0
            else:
                left = i-1
                right = i+1

            #process gaps
            #compute lenght of gap using al-kashi formula
            gap = np.sqrt(objects[left]**2 + objects[right]**2 -2*objects[left]*objects[right]*math.cos(angles[right]-angles[left]))
            if gap < 2:
                temp_objects[i] = min(objects[left], objects[right])
                print(f"filled in gap at angle {angles[i]*180/math.pi}")

        objects[:] = temp_objects[:]

        distMin = 150
        for i, object in enumerate(objects):
                x_obj = object*math.cos(angles[i]+self.arc/2)
                y_obj = object*math.sin(angles[i]+self.arc/2)

                x_goal = goal_dist*math.sin(goal_angle) #reference frame for angle to goal is inverted
                y_goal = goal_dist*math.cos(goal_angle)

                dist_obj2goal = np.sqrt((y_goal-y_obj)**2 + (x_goal-x_obj)**2)

                #check if the dwa will be able to reach that point easily.
                if object < 10: #1. if the d_leave distance found is towards an obstacle, the real achievable distance is dist_obj2goal + the safety margin of the dwa.
                    dist_obj2goal += 1.5


                if dist_obj2goal < distMin:
                    distMin = dist_obj2goal
                    orientation = angles[i]
                    idx = i

        #print(f"orientation of d_leave: {orientation*180/math.pi}")

        return distMin, orientation, idx


    def compute_d_min(self, objects, angles, goal_dist, goal_angle, object_to_avoid, segments):
        distMin = self.d_min
        for i, object in enumerate(objects):
            if segments[i] == object_to_avoid: #only update d_min if we confirmed this is on the boundary of the obstacle
                x_obj = object*math.cos(angles[i]+self.arc/2)
                y_obj = object*math.sin(angles[i]+self.arc/2)

                x_goal = goal_dist*math.sin(goal_angle) #reference frame for angle to goal is inverted
                y_goal = goal_dist*math.cos(goal_angle)

                dist_obj2goal = np.sqrt((y_goal-y_obj)**2 + (x_goal-x_obj)**2)

                if dist_obj2goal < distMin:
                    distMin = dist_obj2goal

        return distMin

    def compute_heuristic(self, object_dist, object_angle, goal_dist, goal_angle):

        x_goal = goal_dist*math.sin(goal_angle) #reference frame for angle to goal is inverted
        y_goal = goal_dist*math.cos(goal_angle)

        if object_dist < goal_dist:
            x_obj = object_dist*math.cos(object_angle+self.arc/2)
            y_obj = object_dist*math.sin(object_angle+self.arc/2)

            dist_uav2obj = object_dist
            dist_obj2goal = np.sqrt((y_goal-y_obj)**2 + (x_goal-x_obj)**2)

            #print(f"angle considered: {object_angle*180/math.pi}")
            #print(f"object distance: {np.round(dist_uav2obj,2)}, obj to goal distance: {np.round(dist_obj2goal,2)} heuristic: {np.round(dist_uav2obj + dist_obj2goal,2)}")

            return max(0.5, dist_uav2obj + dist_obj2goal)
        else: #goal is in front of obstacle
            x_obj = goal_dist*math.cos(object_angle+self.arc/2)
            y_obj = goal_dist*math.sin(object_angle+self.arc/2)

            dist_obj2goal = np.sqrt((y_goal-y_obj)**2 + (x_goal-x_obj)**2)

            #print(f"Heuristic: {dist_obj2goal}")
            return max(0.5, dist_obj2goal)

    def compute_heuristic_verbose(self, object_dist, object_angle, goal_dist, goal_angle):

        if object_dist < goal_dist:
            x_goal = goal_dist*math.sin(goal_angle) #reference frame for angle to goal is inverted
            y_goal = goal_dist*math.cos(goal_angle)

            x_obj = object_dist*math.cos(object_angle+self.arc/2)
            y_obj = object_dist*math.sin(object_angle+self.arc/2)

            dist_uav2obj = object_dist
            dist_obj2goal = np.sqrt((y_goal-y_obj)**2 + (x_goal-x_obj)**2)

            print(f"angle considered: {object_angle*180/math.pi}")
            print(f"object distance: {np.round(dist_uav2obj,2)}, obj to goal distance: {np.round(dist_obj2goal,2)} heuristic: {np.round(dist_uav2obj + dist_obj2goal,2)}")

            return dist_uav2obj + dist_obj2goal
        else: #goal is in front of obstacle
            print(f"Heuristic: {goal_dist}")
            return goal_dist

    def compute_discontinuities(self, objects):
        """receives lidar data, and divides data into continuous segments. returns a list of the tags for each object.
        ex: for objects [1 1.1 1.2 1.3 5 5.1 3.1 3.2 3.1 0.9], will output [0 0 0 0 1 1 2 2 2 0] """

        discontinuity_treshold = 0.9
        diff = []
        diff.append(objects[0]-objects[-1])
        for i in range(len(objects)-1):
            diff.append(objects[i+1]-objects[i])

        segments = [0]*len(objects)
        discontinuities = 0
        for i, delta in enumerate(diff):
            if abs(delta) > discontinuity_treshold:
                discontinuities+=1
            segments[i] = discontinuities

        if abs(diff[0]) < discontinuity_treshold:
            for i, segment in enumerate(segments):
                if segment == discontinuities:
                    segments[i] = 0

        
        return segments

    def find_obstacles(self, sensors):
        """ receives an array of sensors and computes where the obstacles are (discontinuities)""" 
