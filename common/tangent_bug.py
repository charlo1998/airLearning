import math
import numpy as np
import settings


class tangent_bug():
    '''
    implementation of a tangent bug algorithm for path planning with obstacle avoidance.
    '''

    def __init__(self):
        self.arc = 2*math.pi/settings.number_of_sensors #rad
        self.d_leave = 150
        self.d_min = 149
        self.following_boundary = False
        self.done =False
        self.max_dist = 3
        self.previous_dist = 150
        self.previous_obs = [3]*(settings.number_of_sensors+4)



    def predict(self, obs):

        obs = obs[0][0] #flattening the list
        obs[4:] = 100**obs[4:] #reconverting from normalized to real values
        obs[1] = 100**obs[1]

        goal_angle = obs[0]*math.pi #rad
        goal_distance = obs[1]
        x_vel = obs[3]
        y_vel = obs[2]
        sensors = obs[4:]
        angles =  np.arange(-math.pi,math.pi,self.arc)

        objects =[]
        orientations = []
        min_dist = 150
        #create objects list to evaluate obstacles positions, and replace missing values with old observations.
        #any distance greater than the treshold will be ceiled.
        for i, sensor in enumerate(sensors):
            if sensor < 99:
                if sensor >= 66:
                    sensors[i] = self.previous_obs[i]
                objects.append(min(sensors[i], self.max_dist))
                orientations.append(angles[i])

        print(f"sensors: {np.round(sensors,1)}")
        print(f"distances: {np.round(objects,1)}")
        print_angles = [x*180/math.pi for x in orientations]
        print(f"angles: {np.round(print_angles,2)}")

        if self.done:
            self.previous_dist = 150
            self.d_leave = 150
            self.d_min = 149

        if(not self.following_boundary):
            #find direction that minimizes distance to goal
            for i, object in enumerate(objects):
                heuristic = self.compute_heuristic(object, orientations[i], goal_distance, goal_angle)
                if heuristic < min_dist:
                    min_dist = heuristic
                    direction = orientations[i]
            
            print(f"direction: {np.round(direction*180/math.pi,2)}")
            goal = [goal_distance*math.cos(direction), goal_distance*math.sin(direction)]  #drone body frame ref
            #take moving action


            if (min_dist > self.previous_dist):
                self.following_boundary = True
                print(f"Switched to boundary following: min_dist is {min_dist} but previous dist is {self.previous_dist}")

            self.previous_dist = min_dist

        else:
            closest_obstacle = np.argmin(objects)
            tangent = orientations[closest_obstacle]+math.pi/2 #always turning left, other option is to check previous closest point to decide left or right
            goal = [self.max_dist*math.cos(tangent), self.max_dist*math.sin(tangent)]

            print(f"closest obstacle is at angle {orientations[closest_obstacle]*180/math.pi}. Tangent:  {tangent*180/math.pi}")
            #move towards tangent

            self.d_leave = self.compute_d_leave(objects, angles, goal_distance, goal_angle)

            #check if goal reached

            print(f"d_leave: {self.d_leave}  d_min: {self.d_min}")
            

            if (self.done or self.d_leave < self.d_min):
                self.following_boundary = False
                print("switched back to normal path")

            self.d_min = self.compute_d_min(objects, angles, goal_distance, goal_angle)


        self.previous_obs = sensors
        print(f"goal distance: {goal_distance} angle: {goal_angle*180/math.pi}")

        return goal


    def compute_d_leave(self, objects, angles, goal_dist, goal_angle):
        distMin = 150
        for i, object in enumerate(objects):
                x_obj = object*math.cos(angles[i]+self.arc/2)
                y_obj = object*math.sin(angles[i]+self.arc/2)

                x_goal = goal_dist*math.sin(goal_angle) #reference frame for angle to goal is inverted
                y_goal = goal_dist*math.cos(goal_angle)

                dist_obj2goal = np.sqrt((y_goal-y_obj)**2 + (x_goal-x_obj)**2)

                if dist_obj2goal < distMin:
                    distMin = dist_obj2goal

        return distMin


    def compute_d_min(self, objects, angles, goal_dist, goal_angle):
        distMin = self.d_min
        for i, object in enumerate(objects):
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

        x_obj = object_dist*math.cos(object_angle+self.arc/2)
        y_obj = object_dist*math.sin(object_angle+self.arc/2)


        dist_uav2obj = object_dist
        dist_obj2goal = np.sqrt((y_goal-y_obj)**2 + (x_goal-x_obj)**2)

        if (dist_obj2goal < self.d_min):
            self.d_min = dist_obj2goal

        return dist_uav2obj + dist_obj2goal


    def find_obstacles(self, sensors):
        """ receives an array of sensors and computes where the obstacles are (discontinuities)""" 
