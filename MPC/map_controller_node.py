#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from math import atan2, sin
import pandas as pd
from geometry_msgs.msg import Point
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32

from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Float32

from .lookup_steer_angle import LookupSteerAngle
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy


class MAP(Node):
    def __init__(self):
        super().__init__('map_controller')

        # ---------------- PARAMETERS ----------------
        self.declare_parameter('map_path', '/home/autodrive_devkit/src/map_controller/map_controller/waypoints.csv')
        self.declare_parameter('lut_path', '/home/autodrive_devkit/src/on_track_sysid/models/SIM/SIM_pacejka_LUT_lookup_table.csv')
        self.declare_parameter('min_lookahead', 0.5)
        self.declare_parameter('max_lookahead', 3.0)
        self.declare_parameter('min_speed', 0.5)
        self.declare_parameter('max_speed', 7.0)

        self.map_path = self.get_parameter('map_path').value
        self.lut_path = self.get_parameter('lut_path').value
        self.min_lookahead = self.get_parameter('min_lookahead').value
        self.max_lookahead = self.get_parameter('max_lookahead').value
        self.min_speed = self.get_parameter('min_speed').value
        self.max_speed = self.get_parameter('max_speed').value

        # ---------------- LOAD MAP & LUT ----------------
        self.waypoints = self.load_raceline_csv(self.map_path)
        self.steer_lookup = LookupSteerAngle(self.lut_path)

        # ---------------- STATE ----------------
        self.position = None
        self.yaw = 0.0
        self.speed = 0.0
        self.lookahead_distance = None

        # ---------------- ROS INTERFACES ----------------
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # --- POSITION from IPS (Point message) ---
        self.create_subscription(
            Point,
            '/autodrive/f1tenth_1/ips',
            self.ips_callback,
            10
        )

# --- ORIENTATION from IMU ---
        self.create_subscription(
            Imu,
            '/autodrive/f1tenth_1/imu',
            self.imu_callback,
            10
        )

# --- SPEED from AutoDRIVE ---
        self.create_subscription(
            Float32,
            '/autodrive/f1tenth_1/speed',
            self.speed_callback,
            10
        )



        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.throttle_pub = self.create_publisher(Float32, '/autodrive/f1tenth_1/throttle_command', 10)
        self.steer_pub = self.create_publisher(Float32, '/autodrive/f1tenth_1/steering_command', 10)

        self.lookahead_pub = self.create_publisher(Marker, '/lookahead_point', 10)

        self.create_timer(1/40.0, self.control_loop)

        self.get_logger().info(" MAP Controller Running")

    # -------------------------------------------------------
    def load_raceline_csv(self, filename):
        
        df = pd.read_csv(filename,header=None)
        pts = df.iloc[:,0:2].values.astype(float)
        return pts

    # -------------------------------------------------------
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def calc_lookahead_dist(self, speed):
        norm = (speed - self.min_speed) / (self.max_speed - self.min_speed)
        sig = self.sigmoid(norm * 10 - 5)
        return self.min_lookahead + sig * (self.max_lookahead - self.min_lookahead)
    def ips_callback(self, msg):
    # Position only
        self.position = np.array([msg.x, msg.y])

    def imu_callback(self, msg):
        q = msg.orientation
        siny = 2*(q.w*q.z + q.x*q.y)
        cosy = 1 - 2*(q.y*q.y + q.z*q.z)
        self.yaw = atan2(siny, cosy)

    def speed_callback(self, msg):
        self.speed = msg.data

    # -------------------------------------------------------
    @staticmethod
    def nearest_idx(position, waypoints):
        return np.argmin(np.linalg.norm(waypoints - position, axis=1))

    def get_lookahead_point(self, position, Ld):
        idx = self.nearest_idx(position, self.waypoints)
        total = 0.0
        prev = self.waypoints[idx]
        i = idx

        while total < Ld:
            i = (i + 1) % len(self.waypoints)
            pt = self.waypoints[i]
            total += np.linalg.norm(pt - prev)
            prev = pt
        return self.waypoints[i]

    # -------------------------------------------------------
    def odom_callback(self, msg):
        self.position = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
        ])

        q = msg.pose.pose.orientation
        siny = 2*(q.w*q.z + q.x*q.y)
        cosy = 1 - 2*(q.y*q.y + q.z*q.z)
        self.yaw = atan2(siny, cosy)

        self.speed = msg.twist.twist.linear.x
        self.lookahead_distance = self.calc_lookahead_dist(self.speed)
    def control_loop(self):

    # ---------------- SAFETY CHECK ----------------
        if self.position is None or self.speed is None or self.yaw is None:
            self.get_logger().warn("Waiting for position / speed / yaw...")
            return

    # ---------------- DYNAMIC LOOKAHEAD ----------------
        self.lookahead_distance = self.calc_lookahead_dist(self.speed)
        Ld = max(self.lookahead_distance, 0.5)

    # ---------------- GET LOOKAHEAD TARGET ----------------
        try:
            target = self.get_lookahead_point(self.position, Ld)
        except Exception as e:
            self.get_logger().error(f"Lookahead error: {e}")
            return

        if target is None:
            self.get_logger().warn("No lookahead target found")
            return

    # ---------------- PURE PURSUIT GEOMETRY ----------------
        vec = target - self.position
        norm = np.linalg.norm(vec)

        if norm < 1e-6:
            self.get_logger().warn("Lookahead vector too small")
            return

    # Signed angle to target
        eta = np.arcsin(
            np.dot(vec, [-np.sin(self.yaw), np.cos(self.yaw)]) / norm
        )

        v = max(self.speed, 0.5)

    # Desired lateral acceleration (signed)
        lat_acc_des = (2 * v**2 * np.sin(eta)) / Ld

    # ---------------- LUT STEERING (MAGNITUDE + SIGN) ----------------
        try:
            steer_mag = self.steer_lookup(abs(lat_acc_des), v)
            sign = np.sign(eta) if abs(eta) > 1e-3 else 0.0
            steer = sign * steer_mag
        except Exception as e:
            self.get_logger().error(f"LUT error: {e}")
            steer = 0.0

    # ---------------- AUTONOMOUS SPEED PLANNING ----------------
        v_max = 3.0     # max allowed speed
        v_min = 0.8     # minimum cruise speed

    # Slow down more when |eta| is large (corner)
        speed_factor = max(0.0, 1.0 - 1.2 * abs(eta))
        v_des = np.clip(v_max * speed_factor, v_min, v_max)

    # ---------------- THROTTLE CONTROLLER ----------------
        Kp = 0.6
        speed_error = v_des - v
        throttle_cmd = Kp * speed_error + 0.2   # small feedforward bias
        throttle_cmd = float(np.clip(throttle_cmd, 0.0, 1.0))

    # ---------------- COMMAND OUTPUT ----------------
        drive = AckermannDriveStamped()
        drive.drive.steering_angle = float(steer)
        drive.drive.speed = float(v_des)
        self.drive_pub.publish(drive)

        throttle = Float32()
        throttle.data = throttle_cmd
        self.throttle_pub.publish(throttle)

        steer_cmd = Float32()
        steer_cmd.data = float(steer)
        self.steer_pub.publish(steer_cmd)

    # ---------------- DEBUG PRINT ----------------
        self.get_logger().info(
            f"MAP → pos={self.position}, v={v:.2f}, v_des={v_des:.2f}, "
            f"Ld={Ld:.2f}, eta={eta:.3f}, lat_acc={lat_acc_des:.3f}, "
            f"steer={steer:.3f}, thr={throttle_cmd:.2f}"
        )

    # -------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = MAP()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

