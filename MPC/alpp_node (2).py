import numpy as np
import rclpy
import rclpy.logging
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Float32
from geometry_msgs.msg import Point
from sensor_msgs.msg import Imu
from math import atan2, sqrt, pow, sin, exp
import math

import pandas as pd
import time


class AdaptivePurePursuit(Node):
    def __init__(self):
        super().__init__('adaptive_pure_pursuit')

        # ---------- REPLACED /odom with /ips and /imu ----------
        self.ips_sub = self.create_subscription(
            Point, '/autodrive/f1tenth_1/ips', self.ips_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, '/autodrive/f1tenth_1/imu', self.imu_callback, 10
        )
        # -------------------------------------------------------

        self.thr_pub = self.create_publisher(
            Float32, '/autodrive/f1tenth_1/throttle_command', 10)
        self.str_pub = self.create_publisher(
            Float32, '/autodrive/f1tenth_1/steering_command', 10)
        self.goal_pub = self.create_publisher(Marker, '/goal', 10)
        self.cp_pub = self.create_publisher(Marker, '/cp', 10)
        self.race_pub = self.create_publisher(MarkerArray, '/raceline', 10)

        timer = 0.001
        self.timer = self.create_timer(timer, self.publish_control_commands)

        self.max_speed = 0.083
        self.min_speed = 0.06
        self.max_lookahead = 2.75  # 1.7
        self.min_lookahead = 1.75  # 1.3

        self.wheelbase = 0.33
        self.current_quaternion = [0.0, 0.0, 0.0, 1.0]
        self.lookahead_distance = self.min_lookahead
        self.beta = 0.5
        self.path = np.array([])
        self.previous_position = None
        self.previous_deviation = 0
        self.total_area = 0
        self.area_window = []
        self.window_size = 10
        self.position = None
        self.orientation = None
        self.control_velocity = 0.0015
        self.heading_angle = 0.01
        self.yaw = 0.0


        self.current_speed = 0.0
        self.last_ips_time = None

        self.load_raceline_csv(
            # '/home/autodrive_devkit/src/alpp_controller/alpp_controller/waypoints.csv'
            'raceline.csv'
        )

    def load_raceline_csv(self, filename):
        self.path = pd.read_csv(filename, header=None)
        self.path = self.path.iloc[1:, :2].to_numpy()   # take only X and Y
        #self.path[:, [0, 1]] = self.path[:, [1, 0]]
        # self.path = self.path[::-1]

        # for i in range(len(self.path)):
        #     self.path[i, 1] += 0
        #     self.path[i, 0] -= 0

        # rotation_matrix = np.array([[-0.06, 1], [-1, 0]])
        # self.path = np.dot(self.path, rotation_matrix.T)
        print("RACELINE LOADED:", self.path.shape)

    # ------------------ IPS callback ------------------
    def ips_callback(self, msg: Point):
        # msg is geometry_msgs/Point
        self.position = msg


        current_time = self.get_clock().now().nanoseconds * 1e-9

        if self.previous_position is not None and self.last_ips_time is not None:
            dx = self.position.x - self.previous_position.x
            dy = self.position.y - self.previous_position.y
            dt = current_time - self.last_ips_time

            if dt > 0:
                self.current_speed = sqrt(dx*dx + dy*dy) / dt

        self.previous_position = self.position
        self.last_ips_time = current_time

        print("IPS:", msg.x, msg.y)
        self.pos_callback(None)

    # ------------------ IMU callback ------------------
    def imu_callback(self, msg):
    # quick sanity prints to prove the callback fires and updates yaw
        print("✅ IMU CALLBACK TRIGGERED")
        q = msg.orientation

        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)

        self.yaw = atan2(siny_cosp, cosy_cosp)

    # helpful compact output
        print(f"✅ YAW UPDATED: {self.yaw:.4f}  HEAD: [{math.cos(self.yaw):.3f} {math.sin(self.yaw):.3f}]")




    # ------------------ pos_callback ------------------
    def pos_callback(self, msg):
        '''if msg is not None:
            self.position = msg.pose.pose.position
            self.orientation = msg.pose.pose.orientation

            self.current_quaternion = [
                self.orientation.x, self.orientation.y, self.orientation.z, self.orientation.w]
            self.yaw = self.quaternion_to_yaw(self.current_quaternion)

            current_speed = msg.twist.twist.linear.x
            self.update_lookahead(current_speed)
        else:
            current_speed = self.current_speed
            self.update_lookahead(current_speed)'''
        current_speed = self.current_speed

        closest_point, goal_point = self.get_lookahead_point(self.position)
        print("GOAL:", goal_point)

        if goal_point is not None:
            alpha = self.calculate_alpha(self.position, goal_point, self.yaw)
            self.last_alpha = alpha
            self.heading_angle = self.calculate_heading_angle(alpha)


            area = self.calculate_deviation(self.position, closest_point)

            max_velocity_pp = self.calculate_max_velocity_pure_pursuit(
                self.calculate_curvature(alpha))
            min_deviation_pp = self.calculate_min_deviation_pure_pursuit(area)

            self.control_velocity = self.convex_combination(
                max_velocity_pp, min_deviation_pp, current_speed, area)
            self.publish_control_commands()

    def quaternion_to_yaw(self, quaternion):
        qx, qy, qz, qw = quaternion
        siny_cosp = 2*(qw * qz + qx * qy)
        cosy_cosp = 1 - 2*(qy * qy + qz * qz)
        yaw = atan2(siny_cosp, cosy_cosp)
        return yaw
    def sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def update_lookahead(self, speed):
        normalized_speed = (speed - self.min_speed) / \
            (self.max_speed - self.min_speed)
        sigmoid_value = self.sigmoid(normalized_speed * 10 - 5)

        if speed < self.min_speed:
            self.lookahead_distance = self.min_lookahead
        else:
            scaled_lookahead = self.min_lookahead + sigmoid_value * \
                (self.max_lookahead - self.min_lookahead)
            self.lookahead_distance = min(self.max_lookahead, scaled_lookahead)

    def get_lookahead_point(self, position):
        # 1. FIX: Keep floats, do not convert to int
        pts = np.array(self.path, dtype=float)

        dists = np.linalg.norm(pts - np.array([position.x, position.y]), axis=1)
        closest_idx = int(np.argmin(dists))
        closest_point = pts[closest_idx]

        goal_point = None
        heading_vec = np.array([np.cos(self.yaw), np.sin(self.yaw)])

        # 2. FIX: Increase search window size (20 -> 100)
        min_idx = closest_idx + 2
        max_idx = min(closest_idx + 100, len(pts)-1) 

        for i in range(min_idx, max_idx+1):
            point = pts[i]
            vec_car_to_point = np.array([point[0] - position.x, point[1] - position.y])
            dist = np.linalg.norm(vec_car_to_point)

            # Debug print to see what the math is doing
            # print(f"Idx: {i}, Dist: {dist:.2f}, Dot: {np.dot(vec_car_to_point, heading_vec):.2f}")

            if dist > self.lookahead_distance and np.dot(vec_car_to_point, heading_vec) > 0:
                goal_point = point
                # ... (Markers code omitted for brevity, keep your original marker code here) ...
                # ...
                break

        return closest_point, goal_point

    # def get_lookahead_point(self, position):
    #     # robust closest index
    #     # ensure path values are numeric (integers)
    #     pts = np.array(self.path, dtype=float)

    #     # print(np.array([int(position.x), int(position.y)]))
    #     # print(self.path)
    #     dists = np.linalg.norm(pts - np.array([position.x, position.y]), axis=1)
    #     closest_idx = int(np.argmin(dists))
    #     closest_point = pts[closest_idx]

    #     goal_point = None

    # # choose a point ahead (search forward along path), ensure it's also in front of car
    #     heading_vec = np.array([np.cos(self.yaw), np.sin(self.yaw)])
    #     if abs(self.yaw) < 1e-4:
    #         print("❌❌❌ YAW IS ZERO — IMU NOT WORKING ❌❌❌")

    # # choose indices with a small forward margin to avoid immediate neighbors
    #     min_idx = closest_idx + 2
    #     max_idx = min(closest_idx + 100, len(pts)-1)  # search window
    #     print(closest_point)
    #     for i in range(min_idx, max_idx+1):
    #         point = pts[i]
    #         vec_car_to_point = np.array([point[0] - position.x, point[1] - position.y])
    #         dist = np.linalg.norm(vec_car_to_point)
    #         print("DOT:", np.dot(vec_car_to_point, heading_vec),"VEC:", vec_car_to_point,"HEAD:", heading_vec)

            
    #         # require lookahead distance AND that the point is in front of the car
    #         if dist > self.lookahead_distance and np.dot(vec_car_to_point, heading_vec) > 0:
                
    #             goal_point = point
                
    #             # publish markers (kept same as original)
    #             marker = Marker()
    #             marker.header.frame_id = 'world'
    #             marker.header.stamp = self.get_clock().now().to_msg()
    #             marker.type = Marker.SPHERE
    #             marker.action = Marker.ADD
    #             marker.pose.position.x = point[0]
    #             marker.pose.position.y = point[1]
    #             marker.pose.position.z = 0.0
    #             marker.scale.x = 0.1
    #             marker.scale.y = 0.1
    #             marker.scale.z = 0.1
    #             marker.color.a = 1.0
    #             marker.color.r = 1.0
    #             marker.color.g = 0.0
    #             marker.color.b = 0.0

    #             marker2 = Marker()
    #             marker2.header.frame_id = 'world'
    #             marker2.header.stamp = self.get_clock().now().to_msg()
    #             marker2.type = Marker.SPHERE
    #             marker2.action = Marker.ADD
    #             marker2.pose.position.x = closest_point[0]
    #             marker2.pose.position.y = closest_point[1]
    #             marker2.pose.position.z = 0.0
    #             marker2.scale.x = 0.1
    #             marker2.scale.y = 0.1
    #             marker2.scale.z = 0.1
    #             marker2.color.a = 1.0
    #             marker2.color.r = 0.0
    #             marker2.color.g = 0.0
    #             marker2.color.b = 1.0

    #             markerarray = MarkerArray()
    #             for j in range(len(pts)):
    #                 m = Marker()
    #                 m.header.frame_id = 'world'
    #                 m.header.stamp = self.get_clock().now().to_msg()
    #                 m.type = Marker.SPHERE
    #                 m.action = Marker.ADD
    #                 m.id = j
    #                 m.pose.position.x = pts[j][0]
    #                 m.pose.position.y = pts[j][1]
    #                 m.pose.position.z = 0.0
    #                 m.scale.x = 0.1
    #                 m.scale.y = 0.1
    #                 m.scale.z = 0.1
    #                 m.color.a = 1.0
    #                 m.color.r = 0.0
    #                 m.color.g = 1.0
    #                 m.color.b = 0.0
    #                 markerarray.markers.append(m)

    #             self.goal_pub.publish(marker)
    #             self.cp_pub.publish(marker2)
    #             self.race_pub.publish(markerarray)
    #             break

    #     return closest_point, goal_point

    def calculate_alpha(self, position, goal_point, yaw):
        dy = goal_point[1] - position.y
        dx = goal_point[0] - position.x
        local_x = dx*np.cos(-yaw) - dy*np.sin(-yaw)
        local_y = dx*np.sin(-yaw) + dy*np.cos(-yaw)
        alpha = atan2(local_y, local_x)
        return alpha

    def calculate_heading_angle(self, alpha):
        heading_angle = atan2(2 * self.wheelbase *
                              sin(alpha), self.lookahead_distance)
        return heading_angle

    def calculate_curvature(self, alpha):
        curvature = 2 * sin(alpha) / self.lookahead_distance
        return curvature

    def calculate_deviation(self, position, closest_point):
        deviation = sqrt(
            pow(closest_point[0] - position.x, 2) + pow(closest_point[1] - position.y, 2))

        if self.previous_position is not None:
            distance_traveled = sqrt(pow(position.x - self.previous_position.x, 2) +
                                     pow(position.y - self.previous_position.y, 2))
            area_increment = (
                deviation + self.previous_deviation) / 2 * distance_traveled

            self.area_window.append(area_increment)
            if len(self.area_window) > self.window_size:
                self.area_window.pop(0)

            self.total_area = sum(self.area_window)

        self.previous_position = position
        self.previous_deviation = deviation

        return self.total_area

    def calculate_max_velocity_pure_pursuit(self, curvature):
        max_velocity = sqrt(
            1 / abs(curvature)) if curvature != 0 else self.max_speed
        return min(self.max_speed, max_velocity)

    def calculate_min_deviation_pure_pursuit(self, area):
        if area > 0:
            min_deviation_velocity = self.max_speed / (1 + area)
        else:
            min_deviation_velocity = self.max_speed
        return min_deviation_velocity

    def convex_combination(self, max_velocity_pp, min_deviation_pp, current_speed, area):
        self.beta = self.adjust_beta(current_speed, area)

        control_velocity = self.beta * max_velocity_pp + \
            (1 - self.beta) * min_deviation_pp
        curvature = self.calculate_curvature(getattr(self, 'last_alpha', 0.0))
        curv_diff = abs(curvature)
        control_velocity /= (1.0 + 2.0 * abs(curv_diff))

        print(control_velocity)
        return control_velocity

    def adjust_beta(self, current_speed, area):
        if area < 1.0:
            return min(1.0, self.beta + 0.25)
        elif current_speed < self.max_speed * 0.4:
            return max(0.0, self.beta - 0.25)
        return self.beta

    def publish_control_commands(self):
        float1 = Float32()
        float2 = Float32()
        # clamp throttle and steering to safe values
        float1.data = float(np.clip(self.control_velocity, 0.015, self.max_speed))
# steering scale and clamp
        steer_cmd = self.heading_angle * 3.5
        steer_cmd = float(np.clip(steer_cmd, -1.0, 1.0))

        float2.data = steer_cmd

        self.thr_pub.publish(float1)
        self.str_pub.publish(float2)



def main(args=None):
    rclpy.init(args=args)

    adaptive_pure_pursuit = AdaptivePurePursuit()
    rate = adaptive_pure_pursuit.create_rate(0.1)

    try:
        rclpy.spin(adaptive_pure_pursuit)
        rate.sleep()
    except KeyboardInterrupt:
        pass

    adaptive_pure_pursuit.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()


