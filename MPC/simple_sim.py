import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped, Point
from visualization_msgs.msg import Marker
import tf2_ros
import math
import numpy as np
import os

class SimpleSim(Node):
    def __init__(self):
        super().__init__('simple_sim')
        
        # --- NEW: LOAD SPAWN POINT FROM CSV ---
        self.x, self.y = self.load_spawn_point('raceline.csv')
        self.v = 0.0
        self.yaw = self.initial_heading
        
        self.get_logger().info(f"SIMULATOR STARTED. Spawning Car at: X={self.x:.2f}, Y={self.y:.2f}")
        
        # Inputs
        self.accel = 0.0
        self.steer = 0.0
        self.L = 0.33  
        self.dt = 0.05 

        self.sub_drive = self.create_subscription(AckermannDriveStamped, '/drive', self.drive_cb, 10)
        self.pub_odom = self.create_publisher(Odometry, '/odom', 10)
        self.pub_car_viz = self.create_publisher(Marker, '/car_viz', 10) # Added box viz
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        self.timer = self.create_timer(self.dt, self.update_physics)

    def load_spawn_point(self, filename):
        # Defaults if file fails
        start_x, start_y = 0.0, 0.0
        self.initial_heading = 0.0
        
        if not os.path.exists(filename):
            filename = os.path.join(os.getcwd(), filename)
            
        if os.path.exists(filename):
            try:
                data = np.loadtxt(filename, delimiter=',', skiprows=1)
                start_x = data[0, 0]
                start_y = data[0, 1]
                
                # Calculate initial heading based on first two points
                next_x = data[1, 0]
                next_y = data[1, 1]
                self.initial_heading = math.atan2(next_y - start_y, next_x - start_x)
                print(f"Loaded spawn from CSV: {start_x}, {start_y}")
            except:
                print("Failed to read CSV spawn point, using 0,0")
        return start_x, start_y

    def drive_cb(self, msg):
        self.accel = msg.drive.acceleration
        self.steer = msg.drive.steering_angle

    def update_physics(self):
        # Kinematic Bicycle Model
        beta = math.atan(0.5 * math.tan(self.steer))
        self.x += self.v * math.cos(self.yaw + beta) * self.dt
        self.y += self.v * math.sin(self.yaw + beta) * self.dt
        self.v += self.accel * self.dt
        self.yaw += (self.v / self.L) * math.sin(beta) * self.dt
        self.v *= 0.99 # Drag

        self.publish_state()

    def publish_state(self):
        now = self.get_clock().now().to_msg()

        # TF
        t = TransformStamped()
        t.header.stamp = now; t.header.frame_id = 'map'; t.child_frame_id = 'base_link'
        t.transform.translation.x = self.x; t.transform.translation.y = self.y
        q = self.euler_to_quaternion(0, 0, self.yaw)
        t.transform.rotation.x = q[0]; t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]; t.transform.rotation.w = q[3]
        self.tf_broadcaster.sendTransform(t)

        # Odom
        odom = Odometry()
        odom.header.stamp = now; odom.header.frame_id = 'map'; odom.child_frame_id = 'base_link'
        odom.pose.pose.position.x = self.x; odom.pose.pose.position.y = self.y
        odom.pose.pose.orientation = t.transform.rotation
        odom.twist.twist.linear.x = self.v
        self.pub_odom.publish(odom)

    def euler_to_quaternion(self, roll, pitch, yaw):
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        return [qx, qy, qz, qw]

def main():
    rclpy.init()
    node = SimpleSim()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: rclpy.shutdown()

if __name__ == '__main__': main()