#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Wrench
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from sensor_msgs.msg import LaserScan
import numpy as np
import tf
import nav_msgs.msg
import rrt

class CarController:


    def __init__(self, goal):
        rospy.init_node('mycar_controller')
        self.time_last_plan = rospy.Time.now()
        self.pub_rear_left = rospy.Publisher('/mycar/force_rear_left', Wrench, queue_size=10)
        self.pub_rear_right = rospy.Publisher('/mycar/force_rear_right', Wrench, queue_size=10)
        self.pub_front_left = rospy.Publisher('/mycar/force_front_left', Wrench, queue_size=10)
        self.pub_front_right = rospy.Publisher('/mycar/force_front_right', Wrench, queue_size=10)
        self.sub_ground_truth = rospy.Subscriber('/mycar/ground_truth', Odometry,
                self.control_callback)
        self.sub_laser = rospy.Subscriber('/mycar/laser/scan', LaserScan,
                self.planning_callback)
        #publish path and reference point to rviz
        self.pub_path = rospy.Publisher('/mycar/path', Path, queue_size=10)
        self.pub_ref_point = rospy.Publisher('/mycar/ref_point', PoseStamped, queue_size=10)

        self.pos = None
        self.yaw = None
        self.X_goal = rrt.SE2_from_param(goal)    
        self.path = None
        self.last_wp_time = None
        self.last_wp = 0
        self.arrived = False
        self.rrt_calc_time = self.time_last_plan

    def rrt_planner(self, laser_scan, max_iterations):
        pos = self.pos
        yaw = self.yaw
        
        # convert laser_scan to points
        points = []
        a_min = laser_scan.angle_min
        a_max = laser_scan.angle_max
        a_inc = laser_scan.angle_increment
        r_min = laser_scan.range_min
        r_max = laser_scan.range_max
        ranges = laser_scan.ranges

        angle = a_min
        for r in ranges:
            if r <= r_max and r >= r_min and angle < a_max:
                x = np.cos(angle+yaw)*r + pos[0]
                y = np.sin(angle+yaw)*r + pos[1]
                radius = 0.05
                points.append([x,y,radius])
            angle+=a_inc
        max_iterations = max_iterations
        X_start = rrt.SE2_from_param([yaw,pos[0],pos[1]])
        X_goal = self.X_goal
        vehicle_radius = 0.15
        box = [-12,12,-12,12]
        ret =  rrt.rrt(X_start=X_start, X_goal=X_goal, vehicle_radius=vehicle_radius,
            box=box, collision_points=points, plot=True,
            max_iterations=max_iterations, dist_plan=1, tolerance=0.15
        )
        return ret

    def planning_callback(self, laser_scan):
        now = rospy.Time.now()

        # find closest point for basic obstacle avoidance
        a0 = laser_scan.angle_min
        da = laser_scan.angle_increment
        i_min = np.argmin(laser_scan.ranges)
        closest_angle = a0 + da*i_min
        min_dist = laser_scan.ranges[i_min]
        #rospy.loginfo('min dist %f m %f deg', min_dist, np.rad2deg(closest_angle))

        # call long-term planning every 5 seconds
        if (now - self.time_last_plan).to_sec() > 5:
            self.time_last_plan = now
            rospy.loginfo('planning using RRT')
            ret = self.rrt_planner(laser_scan,200)
            self.rrt_calc_time = rospy.Time.now()-self.time_last_plan
            path = ret['path']
            path_full = ret['path_full']
            success = ret['success']

            if success: #reset path and wp counter and timestamp
                self.path = path
                self.last_wp_time = self.time_last_plan+self.rrt_calc_time
                self.last_wp = 0
            else:
                rospy.loginfo('no path reached goal')
                self.path = None
                
            
            #publish planned path to rviz
            msg = Path()
            for p in path_full:
                pose = PoseStamped()
                x, y = p
                z = 0.2
                pose.header.frame_id = 'map'
                pose.pose.position.x = x
                pose.pose.position.y = y
                pose.pose.position.z = z
                msg.poses.append(pose)
            msg.header.frame_id = 'map'
            self.pub_path.publish(msg)

    def control_callback(self, odom):
        # position
        position = odom.pose.pose.position
        self.pos = position

        # orientation
        orientation = odom.pose.pose.orientation
        euler = tf.transformations.euler_from_quaternion(
            (orientation.x, orientation.y, orientation.z, orientation.w))
        roll, pitch, yaw = euler
        self.yaw = yaw

        # rotation and linear rate
        rates = odom.twist.twist.angular
        vel = odom.twist.twist.linear
        yaw_rate = rates.z
        speed = np.sqrt(vel.x**2 + vel.y**2)

        # publish fram for transform tree
        br = tf.TransformBroadcaster()
        br.sendTransform(
            (position.x, position.y, position.z),
            (orientation.x, orientation.y, orientation.z, orientation.w) ,
            rospy.Time.now(),
            "base_link",
            "map")

        drive_torque = 0.1

        X_cur = rrt.SE2_from_param([yaw,position.x,position.y])
        u, R, d = rrt.find_u_R_d(X_cur,self.X_goal)
        if (d <= 0.15):
            self.arrived = True
            self.path = None

        now = rospy.Time.now()
        status = False

        if self.path != None: #path is planned
            X0 = self.path[self.last_wp]
            X1 = self.path[self.last_wp + 1]
            v = rrt.SE2_log(rrt.SE2_inv(X0).dot(X1))
            u_r = 0.2 * np.sign(v[1]) #reference speed
            leg_time = v[1] / u_r
            t = (now-self.last_wp_time).to_sec()/leg_time

            if t >= 1: # switch leg
                self.last_wp_time = now()
                self.last_wp += 1
                if self.last_wp >= len(self.path): #finished current path
                    self.path = None

            Xr = X0.dot(rrt.SE2_exp(v*t))
            vr = rrt.SE2_to_param(Xr)
            dtheta, dx, dy = rrt.SE2_log(rrt.SE2_inv(X_cur).dot(Xr))
            ref_p = PoseStamped()
            
            #publishing reference point on the path
            theta = vr[0]
            ref_p.pose.position.x = vr[1]
            ref_p.pose.position.y = vr[2]
            ref_p.pose.position.z = 0.2
            ref_p.header.frame_id = 'map'
            q = tf.transformations.quaternion_from_euler(0,0,theta)
            ref_p.pose.orientation.x = q[0]
            ref_p.pose.orientation.y = q[1]
            ref_p.pose.orientation.z = q[2]
            ref_p.pose.orientation.w = q[3]
            self.pub_ref_point.publish(ref_p)
            
            # errors
            theta_K = 0.7
            xtrack_K = 0.7
            
            u = np.sqrt(vel.x**2 + vel.y**2)
            dt = u/dx
            omega_r = dtheta/dt
            omega_c = omega_r + theta_K * (dtheta + xtrack_K * dy)
            domega = omega_c - yaw_rate

            # steering PD controller
            steer_kP = 0.707
            steer_kD = 0.707
            
            steer_torque = steer_kP*dtheta + steer_kD*domega

            # speed P controller
            speed_kP = 0.5
            speed_error = u_r - speed

            drive_torque = speed_kP*speed_error

            # saturate torques
            if np.abs(steer_torque) > 0.1:
                steer_torque = np.sign(steer_torque)*0.1
            if np.abs(drive_torque) > 0.1:
                drive_torque = np.sign(drive_torque)*0.1
            # drive
            self.drive(drive_torque,steer_torque)

        else: # no path planned
            u_r = 0 # desired speed is zero for it to stop
            speed_kP = 2
            drive_torque = speed_kP*(u_r - speed)
            


    def drive(self, drive_torque, steer_torque):
        # rear drive
        msg = Wrench()
        msg.force.x = 0
        msg.force.y = 0
        msg.force.z = 0
        msg.torque.x = -drive_torque
        msg.torque.y = 0
        msg.torque.z = 0
        self.pub_rear_left.publish(msg)
        self.pub_rear_right.publish(msg)

        # front left steering
        msg = Wrench()
        msg.force.x = 0
        msg.force.y = 0
        msg.force.z = 0
        msg.torque.x = 0
        msg.torque.y = 0
        msg.torque.z = steer_torque
        self.pub_front_left.publish(msg)

        # front right steering
        msg = Wrench()
        msg.force.x = 0
        msg.force.y = 0
        msg.force.z = 0
        msg.torque.x = 0
        msg.torque.y = 0
        msg.torque.z = steer_torque
        self.pub_front_right.publish(msg)


if __name__ == "__main__":
    goal = [0,0,9.5]
    CarController(goal)
    rospy.spin()
