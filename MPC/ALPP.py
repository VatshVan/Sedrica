import numpy as np
import rclpy
import rclpy.logging
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from math import atan2, sqrt, pow, sin, exp
import pandas as pd
import os

class AdaptivePurePursuit(Node):
    def __init__(self):
        super().__init__('adaptive_pure_pursuit')

        # --- 1. Interface with Simulator ---
        self.pos_sub = self.create_subscription(Odometry, '/odom', self.pos_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        
        # --- 2. Visualization Publishers ---
        self.goal_pub = self.create_publisher(Marker, '/goal', 10)
        self.cp_pub = self.create_publisher(Marker, '/cp', 10)
        self.race_pub = self.create_publisher(MarkerArray, '/raceline', 10)
        
        self.pub_global_track = self.create_publisher(Marker, '/global_track', 10)
        self.pub_history = self.create_publisher(Marker, '/history_path', 10)
        self.pub_lap_text = self.create_publisher(Marker, '/lap_info', 10)

        # --- 3. Parameters ---
        self.max_speed = 5.0      
        self.min_speed = 1.0      
        self.max_lookahead = 3.0  
        self.min_lookahead = 1.0  

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
        
        self.control_velocity = 0.0
        self.heading_angle = 0.0
        self.yaw = 0.0
        
        # FIX: Store current speed globally so we can calculate acceleration
        self.current_speed = 0.0

        # --- 4. Lap Timing & History State ---
        self.lap_count = 0
        self.is_racing = False
        self.lap_start_time = None
        self.history_x = []
        self.history_y = []
        self.points_in_lap = 0

        # --- 5. Load Track ---
        track_file = 'raceline.csv'
        if not os.path.exists(track_file):
            track_file = os.path.join(os.getcwd(), 'raceline.csv')
            
        self.load_raceline_csv(track_file)
        self.get_logger().info("ALPP Node Initialized")

    def load_raceline_csv(self, filename):
        if os.path.exists(filename):
            self.path = pd.read_csv(filename)
            self.path = self.path.to_numpy()[:, :2]
            self.get_logger().info(f"Loaded {len(self.path)} points from {filename}")
        else:
            self.get_logger().error(f"File not found: {filename}")
            theta = np.linspace(0, 2*np.pi, 200)
            self.path = np.vstack((10*np.cos(theta), 6*np.sin(theta))).T

    def sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def update_lookahead(self, speed):
        normalized_speed = (speed - self.min_speed) / (self.max_speed - self.min_speed)
        if self.max_speed == self.min_speed: normalized_speed = 0
        sigmoid_value = self.sigmoid(normalized_speed * 10 - 5)

        if speed < self.min_speed:
            self.lookahead_distance = self.min_lookahead
        else:
            scaled_lookahead = self.min_lookahead + sigmoid_value * (self.max_lookahead - self.min_lookahead)
            self.lookahead_distance = min(self.max_lookahead, scaled_lookahead)

    def pos_callback(self, msg):
        self.position = msg.pose.pose.position
        self.orientation = msg.pose.pose.orientation

        self.current_quaternion = [
            self.orientation.x, self.orientation.y, self.orientation.z, self.orientation.w]
        self.yaw = self.quaternion_to_yaw(self.current_quaternion)

        # Store Speed for P-Controller
        self.current_speed = msg.twist.twist.linear.x

        # 1. Update Lookahead
        self.update_lookahead(self.current_speed)

        # 2. Get Targets & Index
        closest_point, goal_point, closest_idx = self.get_lookahead_point(self.position)

        # LAP LOGIC
        if closest_idx is not None:
            if not self.is_racing and closest_idx > 50:
                self.is_racing = True
                self.lap_start_time = self.get_clock().now()
                self.points_in_lap = 0
                self.get_logger().info(f"Started Lap {self.lap_count}!")
            
            if self.is_racing:
                self.points_in_lap += 1
            
            if self.is_racing and closest_idx < 20 and self.points_in_lap > 200:
                self.finish_lap()

        if goal_point is not None:
            alpha = self.calculate_alpha(self.position, goal_point, self.yaw)
            self.heading_angle = self.calculate_heading_angle(alpha)

            area = self.calculate_deviation(self.position, closest_point)
            curvature = self.calculate_curvature(alpha)
            
            max_velocity_pp = self.calculate_max_velocity_pure_pursuit(curvature)
            min_deviation_pp = self.calculate_min_deviation_pure_pursuit(area)

            self.control_velocity = self.convex_combination(
                max_velocity_pp, min_deviation_pp, self.current_speed, area)
            
            self.publish_control_commands()
        else:
            self.control_velocity = 0.0
            self.heading_angle = 0.0
            self.publish_control_commands()

        # Viz Updates
        self.history_x.append(self.position.x)
        self.history_y.append(self.position.y)
        if len(self.history_x) > 2000:
            self.history_x.pop(0)
            self.history_y.pop(0)

        self.publish_history()
        self.publish_lap_text()
        self.publish_global_track()

    def finish_lap(self):
        current_time = self.get_clock().now()
        elapsed = (current_time - self.lap_start_time).nanoseconds / 1e9
        self.get_logger().info(f"Lap {self.lap_count} Complete! Time: {elapsed:.2f}s")
        
        self.lap_count += 1
        self.points_in_lap = 0
        self.lap_start_time = self.get_clock().now()

        if self.lap_count >= 10:
            self.get_logger().info("BENCHMARK COMPLETE! Stopping.")
            stop_msg = AckermannDriveStamped()
            self.drive_pub.publish(stop_msg)
            raise SystemExit

    def publish_history(self):
        if len(self.history_x) < 2: return
        msg = Marker(); msg.header.frame_id = "map"; msg.id = 500; msg.type = Marker.LINE_STRIP; msg.action = Marker.ADD
        msg.scale.x = 0.08; msg.color.a = 1.0; msg.color.r = 1.0; msg.color.g = 0.0; msg.color.b = 1.0 
        for i in range(len(self.history_x)): 
            msg.points.append(Point(x=self.history_x[i], y=self.history_y[i], z=0.0))
        self.pub_history.publish(msg)

    def publish_lap_text(self):
        msg = Marker(); msg.header.frame_id = "map"; msg.id = 501; msg.type = Marker.TEXT_VIEW_FACING; msg.action = Marker.ADD
        msg.scale.z = 1.5; msg.color.a = 1.0; msg.color.r = 1.0; msg.color.g = 1.0; msg.color.b = 1.0 
        msg.pose.position.x = self.position.x; msg.pose.position.y = self.position.y; msg.pose.position.z = 1.0 
        
        time_str = "0.0s"
        if self.is_racing and self.lap_start_time is not None:
             current = (self.get_clock().now() - self.lap_start_time).nanoseconds / 1e9
             time_str = f"{current:.1f}s"
        
        msg.text = f"LAP: {self.lap_count} | {time_str}"
        self.pub_lap_text.publish(msg)

    def publish_global_track(self):
        msg = Marker(); msg.header.frame_id = "map"; msg.id = 1000; msg.type = Marker.LINE_STRIP; msg.action = Marker.ADD
        msg.scale.x = 0.1; msg.color.a = 1.0; msg.color.b = 1.0 
        for pt in self.path: msg.points.append(Point(x=float(pt[0]), y=float(pt[1]), z=0.0))
        msg.points.append(Point(x=float(self.path[0][0]), y=float(self.path[0][1]), z=0.0))
        self.pub_global_track.publish(msg)

    # --- ALPP MATH HELPER FUNCTIONS ---
    def quaternion_to_yaw(self, quaternion):
        qx, qy, qz, qw = quaternion
        siny_cosp = 2*(qw * qz + qx * qy)
        cosy_cosp = 1 - 2*(qy * qy + qz * qz)
        yaw = atan2(siny_cosp, cosy_cosp)
        return yaw

    def get_lookahead_point(self, position):
        distances = np.sqrt(np.sum((self.path - np.array([position.x, position.y]))**2, axis=1))
        closest_idx = np.argmin(distances)
        closest_point = self.path[closest_idx]

        goal_point = None
        for i in range(closest_idx, len(self.path) + closest_idx):
            idx = i % len(self.path)
            pt = self.path[idx]
            dist = sqrt(pow(pt[0] - position.x, 2) + pow(pt[1] - position.y, 2))
            
            if dist > self.lookahead_distance:
                goal_point = pt
                self.visualize_points(closest_point, goal_point)
                return closest_point, goal_point, closest_idx
                
        return closest_point, self.path[(closest_idx + 5) % len(self.path)], closest_idx

    def visualize_points(self, closest, goal):
        # Goal (Green)
        marker = Marker(); marker.header.frame_id = 'map'; marker.header.stamp = self.get_clock().now().to_msg()
        marker.type = Marker.SPHERE; marker.action = Marker.ADD
        marker.pose.position.x = float(goal[0]); marker.pose.position.y = float(goal[1])
        marker.scale.x = 0.3; marker.scale.y = 0.3; marker.scale.z = 0.3
        marker.color.a = 1.0; marker.color.g = 1.0
        self.goal_pub.publish(marker)

        # Closest (Blue)
        marker2 = Marker(); marker2.header.frame_id = 'map'; marker2.header.stamp = self.get_clock().now().to_msg()
        marker2.type = Marker.SPHERE; marker2.action = Marker.ADD
        marker2.pose.position.x = float(closest[0]); marker2.pose.position.y = float(closest[1])
        marker2.scale.x = 0.3; marker2.scale.y = 0.3; marker2.scale.z = 0.3
        marker2.color.a = 1.0; marker2.color.b = 1.0
        self.cp_pub.publish(marker2)

        if self.race_pub.get_subscription_count() > 0:
            ma = MarkerArray()
            m = Marker()
            m.header.frame_id = 'map'
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.1
            m.color.a = 1.0; m.color.r = 0.0; m.color.g = 1.0
            from geometry_msgs.msg import Point
            for pt in self.path:
                p = Point(); p.x = float(pt[0]); p.y = float(pt[1])
                m.points.append(p)
            ma.markers.append(m)
            self.race_pub.publish(ma)

    def calculate_alpha(self, position, goal_point, yaw):
        dy = goal_point[1] - position.y
        dx = goal_point[0] - position.x
        local_x = dx*np.cos(-yaw) - dy*np.sin(-yaw)
        local_y = dx*np.sin(-yaw) + dy*np.cos(-yaw)
        alpha = atan2(local_y, local_x)
        return alpha

    def calculate_heading_angle(self, alpha):
        return atan2(2 * self.wheelbase * sin(alpha), self.lookahead_distance)

    def calculate_curvature(self, alpha):
        dist = self.lookahead_distance if self.lookahead_distance > 0.01 else 0.01
        return 2 * sin(alpha) / dist

    def calculate_deviation(self, position, closest_point):
        deviation = sqrt(pow(closest_point[0] - position.x, 2) + pow(closest_point[1] - position.y, 2))
        if self.previous_position is not None:
            distance_traveled = sqrt(pow(position.x - self.previous_position.x, 2) + pow(position.y - self.previous_position.y, 2))
            area_increment = (deviation + self.previous_deviation) / 2 * distance_traveled
            self.area_window.append(area_increment)
            if len(self.area_window) > self.window_size: self.area_window.pop(0)
            self.total_area = sum(self.area_window)
        self.previous_position = position
        self.previous_deviation = deviation
        return self.total_area

    def calculate_max_velocity_pure_pursuit(self, curvature):
        if curvature == 0: return self.max_speed
        max_velocity = sqrt(1 / abs(curvature)) 
        return min(self.max_speed, max_velocity)

    def calculate_min_deviation_pure_pursuit(self, area):
        if area > 0: return self.max_speed / (1 + area)
        return self.max_speed

    def convex_combination(self, max_velocity_pp, min_deviation_pp, current_speed, area):
        self.beta = self.adjust_beta(current_speed, area)
        control_velocity = self.beta * max_velocity_pp + (1 - self.beta) * min_deviation_pp
        curvature_val = self.calculate_curvature(self.heading_angle)
        control_velocity /= exp(2.4698 * (abs(curvature_val) ** 0.75))
        return control_velocity

    def adjust_beta(self, current_speed, area):
        if area < 1.0: return min(1.0, self.beta + 0.25)
        elif current_speed < self.max_speed * 0.4: return max(0.0, self.beta - 0.25)
        return self.beta

    def publish_control_commands(self):
        if self.control_velocity is None or self.heading_angle is None: return
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = "base_link"
        
        drive_msg.drive.steering_angle = self.heading_angle 
        drive_msg.drive.speed = float(self.control_velocity)
        
        # --- FIX: P-Controller for Acceleration ---
        # The simulator uses physics, so it needs acceleration to move.
        # Calculate Error: Target Speed - Current Speed
        speed_error = self.control_velocity - self.current_speed
        
        # Proportional Gain (Kp)
        kp = 2.0 
        drive_msg.drive.acceleration = kp * speed_error
        
        self.drive_pub.publish(drive_msg)

def main(args=None):
    rclpy.init(args=args)
    adaptive_pure_pursuit = AdaptivePurePursuit()
    try: rclpy.spin(adaptive_pure_pursuit)
    except KeyboardInterrupt: pass
    adaptive_pure_pursuit.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__': main()