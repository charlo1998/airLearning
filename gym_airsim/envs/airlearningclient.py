import airsim # sudo pip install airsim
import numpy as np
import math
import time
import cv2
import settings

from PIL import Image
from pylab import array, uint8, arange

import msgs

#performances:
import time


class AirLearningClient(airsim.MultirotorClient):
    def __init__(self):

        # connect to the AirSim simulator
        self.client = airsim.MultirotorClient(settings.ip)
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        self.home_pos = self.client.getPosition()
        self.home_ori = self.client.getOrientation()
        self.z = -4

    def goal_direction(self, goal, pos):

        pitch, roll, yaw = self.client.getPitchRollYaw()
        yaw = math.degrees(yaw)

        pos_angle = math.atan2(goal[1] - pos[1], goal[0] - pos[0])
        pos_angle = math.degrees(pos_angle) % 360

        track = math.radians(pos_angle - yaw)
        track = ((math.degrees(track) - 180) % 360) - 180

        return track

    def getConcatState(self, track, goal): #for future perf tests, track was recmputed here with get get_drone_pos instead of being passed like now
        distances = self.get_laser_state()

        #ToDo: Add RGB, velocity etc
        if(settings.position): #This is for ablation purposes
            pos = self.get_distance(goal)

        if(settings.velocity): #This is for ablation purposes
            vel = self.drone_velocity()

        if(settings.position and settings.velocity):
            concat_state = np.concatenate((distances, pos, vel), axis = None)
        elif(settings.position):
            concat_state = np.concatenate((distances, pos), axis = None)
        elif(settings.velocity):
            concat_state = np.concatenate((distances, pos), axis = None)
        else:
            concat_state = distances

        concat_state_shape = concat_state.shape
        concat_state = concat_state.reshape(1, concat_state_shape[0])
        concat_state = np.expand_dims(concat_state, axis=0)

        return concat_state

    def getScreenGrey(self):
        responses = self.client.simGetImages([airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)])
        response = responses[0]
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        if((responses[0].width != 0 or responses[0].height != 0)):
            img_rgba = img1d.reshape(response.height, response.width, 4)
            rgb = cv2.cvtColor(img_rgba, cv2.COLOR_BGRA2BGR)
            grey = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        else:
            print("Something bad happened! Restting AirSim!")
            self.AirSim_reset()
            grey = np.ones(144,256)
        return grey

    def getScreenRGB(self):
        responses = self.client.simGetImages([airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)])
        response = responses[0]
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        if ((responses[0].width != 0 or responses[0].height != 0)):
            img_rgba = img1d.reshape(response.height, response.width, 4)
            rgb = cv2.cvtColor(img_rgba, cv2.COLOR_BGRA2BGR)
        else:
            print("Something bad happened! Restting AirSim!")
            self.AirSim_reset()
            rgb = np.ones(144, 256, 3)
        return rgb

    def getScreenDepthVis(self, track):

        responses = self.client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.DepthPerspective, True, False)])
        #print(f"received {len(responses)} images from client")
        #responses = self.client.simGetImages([airsim.ImageRequest("0",
        #airsim.ImageType.DepthVis,True, False)])

        if(responses == None):
            print("Camera is not returning image!")
            print("Image size:" + str(responses[0].height) + "," + str(responses[0].width))
        else:
            img1d = np.array(responses[0].image_data_float, dtype=np.float)

        img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
        if((responses[0].width != 0 or responses[0].height != 0)):
            img2d = np.reshape(img1d, (responses[0].height, responses[0].width))

        else:
            print("Something bad happened! Restting AirSim!")
            self.AirSim_reset()
            img2d = np.ones((144, 256))

        image = np.invert(np.array(Image.fromarray(img2d.astype(np.uint8), mode='L')))

        factor = 10
        maxIntensity = 255.0  # depends on dtype of image data

        # Decrease intensity such that dark pixels become much darker, bright
        # pixels become slightly dark
        newImage1 = (maxIntensity) * (image / maxIntensity) ** factor
        newImage1 = array(newImage1, dtype=uint8)

        # small = cv2.resize(newImage1, (0,0), fx=0.39, fy=0.38)
        small = cv2.resize(newImage1, (0, 0), fx=1.0, fy=1.0)

        cut = small[20:40, :]
        # print(cut.shape)

        info_section = np.zeros((10, small.shape[1]), dtype=np.uint8) + 255
        info_section[9, :] = 0

        line = np.int((((track - -180) * (small.shape[1] - 0)) / (180 - -180)) + 0)
        '''
        print("\n")
        print("Track:"+str(track))
        print("\n")
        print("(Track - -180):"+str(track - -180))
        print("\n")
        print("Num:"+str((track - -180)*(100-0)))
        print("Num_2:"+str((track+180)*(100-0)))
        print("\n")
        print("Den:"+str(180 - -180))
        print("Den_2:"+str(180+180))
        print("line:"+str(line))
        '''
        if line != (0 or small.shape[1]):
            info_section[:, line - 1:line + 2] = 0
        elif line == 0:
            info_section[:, 0:3] = 0
        elif line == small.shape[1]:
            info_section[:, info_section.shape[1] - 3:info_section.shape[1]] = 0

        total = np.concatenate((info_section, small), axis=0)
        #cv2.imwrite("test.png",total)
        # cv2.imshow("Test", total)
        # cv2.waitKey(0)
        #total = np.reshape(total, (154,256))
        return total

    def drone_pos(self):
        x = self.client.getPosition().x_val
        y = self.client.getPosition().y_val
        z = self.client.getPosition().z_val

        return np.array([x, y, z])

    def drone_velocity(self):
        v_x = self.client.getVelocity().x_val
        v_y = self.client.getVelocity().y_val
        v_z = self.client.getVelocity().z_val

        return np.array([v_x, v_y, v_z])

    def get_distance(self, goal):
        now = self.client.getPosition()
        xdistance = (goal[0] - now.x_val)
        ydistance = (goal[1] - now.y_val)
        euclidean = np.sqrt(np.power(xdistance,2) + np.power(ydistance,2))
        angle = self.goal_direction(goal, [now.x_val, now.y_val])

        return np.array([angle, euclidean])

    def get_velocity(self):
        return np.array([self.client.get_velocity().x_val, self.client.get_velocity().y_val, self.client.get_velocity().z_val])

    #def get_laser_pointer(self):
    #    distance_sensor_data = self.client.getDistanceSensorData(distance_sensor_name = "distance1", vehicle_name = "drone1")
    #    print(distance_sensor_data)

    def get_laser_state(self):
        """
        lidar parameters are set in the settings.json files in documents/airsim. they are the following:
        "Lidarfront": {
       		"SensorType": 6, #the type corresponding to the LiDaR sensor
        	"Enabled" : true,
        	"NumberOfChannels": 3, #the number of lasers (we need a laser for each horizontal swipe to have vertical FOV)
        	"RotationsPerSecond": 20, #frequency of sensing
    		"PointsPerSecond": 100000, #the number of points taken in an horizontal swipe of the lidar (1 rotation).
                #PointsPerSecond is not affected by the FOV, so anything outside the FOV seems to be wasted.
    		"X": 0, "Y": 0, "Z": -1, #position relative to the vehicule
       		"Roll": 0, "Pitch": 0, "Yaw" : 0,   orientation relative to the vehicule
    		"VerticalFOVUpper": 10,
    		"VerticalFOVLower": -30,
       		"HorizontalFOVStart": -5,
       		"HorizontalFOVEnd": 5,
       		"DrawDebugPoints": true, #set to true to see in unreal what is being observed
       		"DataFrame": "SensorLocalFrame"
    		}
        """

        ## -- laser ranger -- ##
        lidarDatafront = self.client.getLidarData(lidar_name="Lidarfront",vehicle_name="Drone1")
        lidarDataleft = self.client.getLidarData(lidar_name="Lidarleft",vehicle_name="Drone1")
        lidarDataright = self.client.getLidarData(lidar_name="Lidarright",vehicle_name="Drone1")
        lidarDataback = self.client.getLidarData(lidar_name="Lidarback",vehicle_name="Drone1")
        
        
        front = self.process_lidar(lidarDatafront.point_cloud)
        left = self.process_lidar(lidarDataleft.point_cloud)
        right = self.process_lidar(lidarDataright.point_cloud)
        back = self.process_lidar(lidarDataback.point_cloud)

        output = [front, left, right, back]
        #print(output)

        return np.array(output)   

    def process_lidar(self, lidarPointcloud):
        """
        processes a lidar point clouds into coordinates and finds closest point.
        """
        points = np.array(lidarPointcloud, dtype=np.dtype('f4'))
        if not (points.shape[0] == 1):
            points = np.reshape(points, (int(points.shape[0]/3), 3))
        else:
            print("lidar not seeing anything ?!")
            return 0

        X = points[:,0]
        Y = points[:,1]
        Z = points[:,2]
        
        distance = min(np.sqrt(X**2+Y**2))

        return distance

    def AirSim_reset(self):
        self.client.reset()
        time.sleep(0.2)
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        time.sleep(0.2)
        self.client.moveByVelocityAsync(0, 0, -5, 2, drivetrain=0, vehicle_name='').join()
        self.client.moveByVelocityAsync(0, 0, 0, 1, drivetrain=0, vehicle_name='').join()

    def unreal_reset(self):
        self.client, connection_established = self.client.resetUnreal()
        return connection_established




    #----------------actions---------------------------
    def take_continious_action(self, action):

        if(msgs.algo == 'DDPG'):
            pitch = action[0]
            roll = action[1]
            throttle = action[2]
            yaw_rate = action[3]
            duration = action[4]
            self.client.moveByAngleThrottleAsync(pitch, roll, throttle, yaw_rate, duration, vehicle_name='').join()
        else:
            #pitch = np.clip(action[0], -0.261, 0.261)
            #roll = np.clip(action[1], -0.261, 0.261)
            #yaw_rate = np.clip(action[2], -3.14, 3.14)
            if(settings.move_by_velocity):
                vx = np.clip(action[0], -1.0, 1.0)
                vy = np.clip(action[1], -1.0, 1.0)
                #print("Vx, Vy--------------->"+str(vx)+", "+ str(vy))
                #self.client.moveByAngleZAsync(float(pitch), float(roll), -6,
                #float(yaw_rate), settings.duration_ppo).join()
                self.client.moveByVelocityZAsync(float(vx), float(vy), -6, 0.5, 1, yaw_mode=airsim.YawMode(True, 0)).join()
            elif(settings.move_by_position):
                pos = self.drone_pos()
                delta_x = np.clip(action[0], -1, 1)
                delta_y = np.clip(action[1], -1, 1)
                self.client.moveToPositionAsync(float(delta_x + pos[0]), float(delta_y + pos[1]), -6, 0.9, yaw_mode=airsim.YawMode(False, 0)).join()
        collided = (self.client.getMultirotorState().trip_stats.collision_count > 0)
        collided = (self.client.getMultirotorState().trip_stats.collision_count > 0)

        return collided

        #Todo : Stabilize drone
        #self.client.moveByAngleThrottleAsync(0, 0,1,0,2).join()

        #TODO: Get the collision info and use that to reset the simuation.
        #TODO: Put some sleep in between the calls so as not to crash on the
        #same lines as DQN
    def straight(self, speed, duration):
        pitch, roll, yaw = self.client.getPitchRollYaw()
        vx = math.cos(yaw) * speed
        vy = math.sin(yaw) * speed
        self.client.moveByVelocityZAsync(vx, vy, self.z, duration, 1).join()
        start = time.time()
        return start, duration

    def backup(self, speed, duration):
        pitch, roll, yaw = self.client.getPitchRollYaw()
        vx = math.cos(yaw) * speed * -0.5
        vy = math.sin(yaw) * speed * -0.5
        self.client.moveByVelocityZAsync(-vx, -vy, self.z, duration, 0).join()
        start = time.time()
        return start, duration

    def yaw_right(self, rate, duration):
        self.client.rotateByYawRateAsync(rate, duration).join()
        start = time.time()
        return start, duration

    def yaw_left(self, rate, duration):
        self.client.rotateByYawRateAsync(-rate, duration).join()
        start = time.time()
        return start, duration

    def pitch_up(self, duration):
        self.client.moveByVelocityZAsync(0.5,0,1,duration,0).join()
        start = time.time()
        return start, duration

    def pitch_down(self, duration):
        self.client.moveByVelocityZAsync(0.5,0.5,self.z,duration,0).join()
        start = time.time()
        return start, duration

    def move_forward_Speed(self, speed_x=0.5, speed_y=0.5, duration=0.5):
        pitch, roll, yaw = self.client.getPitchRollYaw()
        vel = self.client.getVelocity()
        vx = math.cos(yaw) * speed_x - math.sin(yaw) * speed_y
        vy = math.sin(yaw) * speed_x + math.cos(yaw) * speed_y
        if speed_x <= 0.01:
            drivetrain = 0
            #yaw_mode = YawMode(is_rate= True, yaw_or_rate = 0)
        elif speed_x > 0.01:
            drivetrain = 0
            #yaw_mode = YawMode(is_rate= False, yaw_or_rate = 0)
        self.client.moveByVelocityZAsync(vx = (vx + vel.x_val) / 2 ,
                             vy = (vy + vel.y_val) / 2 , #do this to try and smooth the movement
                             z = self.z,
                             duration = duration,
                             drivetrain = drivetrain,).join()
        start = time.time()
        return start, duration

    def take_discrete_action(self, action):

               # check if copter is on level cause sometimes he goes up without a
       # reason
        """
        x = 0
        while self.client.getPosition().z_val < -7.0:
            self.client.moveToZAsync(-6, 3).join()
            time.sleep(1)
            print(self.client.getPosition().z_val, "and", x)
            x = x + 1
            if x > 10:
                return True

        start = time.time()
        duration = 0
        """

        """
        discretization of the action duration: (e^((action//root)/(root-1))*15 - 14)*mv_fw_dur
        the constants 10 and 9 are chosen to fix the range of the function to about [1,25]*mv_fw_dur.
        the actual duration is then interpolated in an exponential function that passes through these boundaries.
        """
        root = np.sqrt(settings.action_discretization)
        
        if action < settings.action_discretization * 1 :
            #go straight
            start, duration = self.straight(settings.mv_fw_spd_5/((action % root)+1), settings.mv_fw_dur*(np.exp((action//root)/(root-1))*15 - 14))
        elif action < settings.action_discretization * 2:
            #go back
            action = action % settings.action_discretization
            start, duration = self.backup(settings.mv_fw_spd_5/((action % root)+1), settings.mv_fw_dur*(np.exp((action//root)/(root-1))*15 - 14))
        elif action < settings.action_discretization * 3:
            #turn right
            action = action % settings.action_discretization
            start, duration = self.yaw_right(settings.yaw_rate_1_1/((action % root)**2+1), settings.rot_dur*(np.exp((action//root)/(root-1))*15 - 14))
        elif action < settings.action_discretization * 4:
            #turn left
            action = action % settings.action_discretization
            start, duration = self.yaw_right(settings.yaw_rate_2_1/((action % root)**2+1), settings.rot_dur*(np.exp((action//root)/(root-1))*15 - 14))

        #go diagonally. this was removed as the drone can just turn then go straight
        #if action == 5:
        #    start, duration = self.move_forward_Speed(settings.mv_fw_spd_5, settings.mv_fw_spd_5, settings.mv_fw_dur*3)

        collided = (self.client.getMultirotorState().trip_stats.collision_count > 0)

        return collided

