import numpy as np
import rclpy
import rclpy.logging
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped # CHANGED: Import Ackermann
from math import atan2, sqrt, pow, sin, exp
import pandas as pd
import os

class AdaptivePurePursuit(Node):
    def __init__(self):
        super().__init__('adaptive_pure_pursuit')

        # --- 1. Interface with Simulator ---
        self.pos_sub = self.create_subscription(Odometry, '/odom', self.pos_callback, 10)
        
        # CHANGED: Publish to /drive instead of separate topics
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        
        # Viz publishers
        self.goal_pub = self.create_publisher(Marker, '/goal', 10)
        self.cp_pub = self.create_publisher(Marker, '/cp', 10)
        self.race_pub = self.create_publisher(MarkerArray, '/raceline', 10)

        # --- 2. Parameters (Tuned for Benchmark) ---
        # Increased speeds to match LMPC benchmark capabilities
        self.max_speed = 5.0      # Was 0.083
        self.min_speed = 1.0      # Was 0.06
        self.max_lookahead = 3.0  # Dynamic lookahead max
        self.min_lookahead = 1.0  # Dynamic lookahead min

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

        # --- 3. Load Track (Raw coordinates) ---
        # Checks local directory first to match LMPC setup
        track_file = 'raceline.csv'
        if not os.path.exists(track_file):
            track_file = os.path.join(os.getcwd(), 'raceline.csv')
            
        self.load_raceline_csv(track_file)
        
        self.get_logger().info("ALPP Node Initialized")

    def load_raceline_csv(self, filename):
        if os.path.exists(filename):
            self.path = pd.read_csv(filename)
            self.path = self.path.to_numpy()[:, :2] # Ensure we just get X,Y
            
            # CHANGED: REMOVED Manual Offsets/Rotations.
            # The simulator spawns the car at the raw CSV coordinates.
            # If we shift the path here, the car will spawn far away from the path.
            
            self.get_logger().info(f"Loaded {len(self.path)} points from {filename}")
        else:
            self.get_logger().error(f"File not found: {filename}")
            # Fallback ellipse if file missing
            theta = np.linspace(0, 2*np.pi, 200)
            self.path = np.vstack((10*np.cos(theta), 6*np.sin(theta))).T

    def sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def update_lookahead(self, speed):
        normalized_speed = (speed - self.min_speed) / (self.max_speed - self.min_speed)
        # Avoid division by zero or extreme values if max/min are close
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

        current_speed = msg.twist.twist.linear.x

        self.update_lookahead(current_speed)

        closest_point, goal_point = self.get_lookahead_point(self.position)

        if goal_point is not None:
            alpha = self.calculate_alpha(self.position, goal_point, self.yaw)
            self.heading_angle = self.calculate_heading_angle(alpha)

            area = self.calculate_deviation(self.position, closest_point)

            curvature = self.calculate_curvature(alpha)
            max_velocity_pp = self.calculate_max_velocity_pure_pursuit(curvature)
            min_deviation_pp = self.calculate_min_deviation_pure_pursuit(area)

            self.control_velocity = self.convex_combination(
                max_velocity_pp, min_deviation_pp, current_speed, area)
            
            self.publish_control_commands()
        else:
            # Emergency Stop if lost
            self.control_velocity = 0.0
            self.heading_angle = 0.0
            self.publish_control_commands()

    def quaternion_to_yaw(self, quaternion):
        qx, qy, qz, qw = quaternion
        siny_cosp = 2*(qw * qz + qx * qy)
        cosy_cosp = 1 - 2*(qy * qy + qz * qz)
        yaw = atan2(siny_cosp, cosy_cosp)
        return yaw

    def get_lookahead_point(self, position):
        # Find closest point
        distances = np.sqrt(np.sum((self.path - np.array([position.x, position.y]))**2, axis=1))
        closest_idx = np.argmin(distances)
        closest_point = self.path[closest_idx]

        # Find Goal Point (Lookahead)
        goal_point = None
        
        # Search forward from closest point
        for i in range(closest_idx, len(self.path) + closest_idx):
            idx = i % len(self.path) # Wrap around
            pt = self.path[idx]
            dist = sqrt(pow(pt[0] - position.x, 2) + pow(pt[1] - position.y, 2))
            
            if dist > self.lookahead_distance:
                goal_point = pt
                self.visualize_points(closest_point, goal_point)
                return closest_point, goal_point
                
        # Fallback if loop logic fails (rare)
        return closest_point, self.path[(closest_idx + 5) % len(self.path)]

    def visualize_points(self, closest, goal):
        # Visualize Goal (Green)
        marker = Marker()
        marker.header.frame_id = 'map' # Changed from world to map for sim compatibility
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.type = Marker.SPHERE; marker.action = Marker.ADD
        marker.pose.position.x = float(goal[0]); marker.pose.position.y = float(goal[1])
        marker.scale.x = 0.3; marker.scale.y = 0.3; marker.scale.z = 0.3
        marker.color.a = 1.0; marker.color.g = 1.0
        self.goal_pub.publish(marker)

        # Visualize Closest (Blue)
        marker2 = Marker()
        marker2.header.frame_id = 'map'
        marker2.header.stamp = self.get_clock().now().to_msg()
        marker2.type = Marker.SPHERE; marker2.action = Marker.ADD
        marker2.pose.position.x = float(closest[0]); marker2.pose.position.y = float(closest[1])
        marker2.scale.x = 0.3; marker2.scale.y = 0.3; marker2.scale.z = 0.3
        marker2.color.a = 1.0; marker2.color.b = 1.0
        self.cp_pub.publish(marker2)

        # Visualize Path (Green Line Strip)
        # Only publish once or low freq to save bandwidth usually, but here is fine
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
        heading_angle = atan2(2 * self.wheelbase * sin(alpha), self.lookahead_distance)
        return heading_angle

    def calculate_curvature(self, alpha):
        # Avoid division by zero
        dist = self.lookahead_distance if self.lookahead_distance > 0.01 else 0.01
        curvature = 2 * sin(alpha) / dist
        return curvature

    def calculate_deviation(self, position, closest_point):
        deviation = sqrt(pow(closest_point[0] - position.x, 2) + pow(closest_point[1] - position.y, 2))

        if self.previous_position is not None:
            distance_traveled = sqrt(pow(position.x - self.previous_position.x, 2) +
                                     pow(position.y - self.previous_position.y, 2))
            area_increment = (deviation + self.previous_deviation) / 2 * distance_traveled

            self.area_window.append(area_increment)
            if len(self.area_window) > self.window_size:
                self.area_window.pop(0)

            self.total_area = sum(self.area_window)

        self.previous_position = position
        self.previous_deviation = deviation
        return self.total_area

    def calculate_max_velocity_pure_pursuit(self, curvature):
        if curvature == 0: return self.max_speed
        max_velocity = sqrt(1 / abs(curvature)) 
        return min(self.max_speed, max_velocity)

    def calculate_min_deviation_pure_pursuit(self, area):
        if area > 0:
            min_deviation_velocity = self.max_speed / (1 + area)
        else:
            min_deviation_velocity = self.max_speed
        return min_deviation_velocity

    def convex_combination(self, max_velocity_pp, min_deviation_pp, current_speed, area):
        self.beta = self.adjust_beta(current_speed, area)
        control_velocity = self.beta * max_velocity_pp + (1 - self.beta) * min_deviation_pp
        
        # Original damping logic
        curvature_val = self.calculate_curvature(self.heading_angle)
        curv_diff = abs(curvature_val)
        control_velocity /= exp(2.4698 * (abs(curv_diff) ** 0.75))
        
        return control_velocity

    def adjust_beta(self, current_speed, area):
        if area < 1.0:
            return min(1.0, self.beta + 0.25)
        elif current_speed < self.max_speed * 0.4:
            return max(0.0, self.beta - 0.25)
        return self.beta

    def publish_control_commands(self):
        if self.control_velocity is None or self.heading_angle is None:
            return

        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = "base_link"
        
        # CHANGED: Removed the * 2.4 multiplier. 
        # For a kinematic sim, we generally want radians directly.
        # If your specific robot needs that gain, add it back.
        drive_msg.drive.steering_angle = self.heading_angle 
        
        drive_msg.drive.speed = float(self.control_velocity)
        drive_msg.drive.acceleration = 0.0 
        
        self.drive_pub.publish(drive_msg)

def main(args=None):
    rclpy.init(args=args)
    adaptive_pure_pursuit = AdaptivePurePursuit()
    try:
        rclpy.spin(adaptive_pure_pursuit)
    except KeyboardInterrupt:
        pass
    adaptive_pure_pursuit.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()