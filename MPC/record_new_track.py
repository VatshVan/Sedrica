import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
import csv
import math
import os

class WaypointLogger(Node):
    def __init__(self):
        super().__init__('waypoint_logger')
        # CHANGE THIS IF NEEDED to '/autodrive/f1tenth_1/ips'
        self.create_subscription(Point, '/autodrive/f1tenth_1/ips', self.save_pose, 10)

        self.file = open('raceline.csv', 'w')
        self.writer = csv.writer(self.file)
        # Writing 4 columns to match your code's expectation
        self.writer.writerow(['x', 'y', 'w_right', 'w_left']) 
        self.get_logger().info("Recording... Drive a lap!")
        self.prev_pose = None

    def save_pose(self, msg):
        x, y = msg.x, msg.y

        # Only record if we moved 10cm
        if self.prev_pose is None or math.hypot(x-self.prev_pose[0], y-self.prev_pose[1]) > 0.1:
            # Saving X, Y, and default widths (2.0m left, 2.0m right)
            self.writer.writerow([x, y, 2.0, 2.0])
            self.prev_pose = (x, y)
            print(f"Recorded: {x:.2f}, {y:.2f}")

def main():
    rclpy.init()
    node = WaypointLogger()
    try: rclpy.spin(node)
    except KeyboardInterrupt:
        node.file.close()
        print("Track Saved to raceline.csv!")

if __name__ == '__main__': main()