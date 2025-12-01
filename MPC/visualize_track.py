import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import numpy as np
import math
import os

# --- SETTINGS ---
FILENAME = 'raceline_2.csv'
FRAME_ID = 'world'  # or 'world' depending on your config

class TrackVisualizer(Node):
    def __init__(self):
        super().__init__('track_visualizer')
        
        # Publishers
        self.pub_center = self.create_publisher(Marker, '/viz_track_center', 10)#, qos_profile=rclpy.qos.qos_profile_sensor_data)
        self.pub_walls = self.create_publisher(Marker, '/viz_track_walls', 10)#, qos_profile=rclpy.qos.qos_profile_sensor_data)
        
        self.path = None
        self.widths = None
        
        self.load_track()
        
        # Publish every 1 second (Low frequency is fine for static map)
        self.create_timer(1.0, self.publish_map)
        self.get_logger().info("Track Visualizer Started. Open RViz!")

    def load_track(self):
        if not os.path.exists(FILENAME):
            self.get_logger().error(f"File {FILENAME} not found!")
            return

        try:
            # Load CSV (Skipping header)
            data = np.loadtxt(FILENAME, delimiter=',', skiprows=1)
            
            # Handle 3-Column (Symmetric) vs 4-Column (Left/Right)
            if data.shape[1] == 3:
                self.path = data[:, :2]
                w = data[:, 2].reshape(-1, 1)
                self.widths = np.hstack((w, w))
            elif data.shape[1] >= 4:
                self.path = data[:, :2]
                self.widths = data[:, 2:4]
            else:
                self.path = data[:, :2]
                self.widths = np.ones((len(data), 2)) * 2.0
                
            self.get_logger().info(f"Loaded {len(self.path)} waypoints.")
            
        except Exception as e:
            self.get_logger().error(f"Failed to load CSV: {e}")

    def publish_map(self):
        if self.path is None: return

        # --- 1. CENTER LINE (Green Strip) ---
        m_center = Marker()
        m_center.header.frame_id = FRAME_ID
        m_center.header.stamp = self.get_clock().now().to_msg()
        m_center.id = 0
        m_center.type = Marker.LINE_STRIP
        m_center.action = Marker.ADD
        m_center.scale.x = 0.15  # Line width
        m_center.color.a = 1.0; m_center.color.g = 1.0  # Green
        
        for p in self.path:
            m_center.points.append(Point(x=p[0], y=p[1], z=0.0))
            
        self.pub_center.publish(m_center)

        # --- 2. WALL BOUNDARIES (White Lines) ---
        m_walls = Marker()
        m_walls.header.frame_id = FRAME_ID
        m_walls.header.stamp = self.get_clock().now().to_msg()
        m_walls.id = 1
        m_walls.type = Marker.LINE_LIST
        m_walls.action = Marker.ADD
        m_walls.scale.x = 0.1
        m_walls.color.a = 1.0; m_walls.color.r = 1.0; m_walls.color.g = 1.0; m_walls.color.b = 1.0 # White

        for i in range(len(self.path) - 1):
            x, y = self.path[i]
            x_next, y_next = self.path[i+1]
            
            # Calculate Heading
            dx = x_next - x
            dy = y_next - y
            yaw = math.atan2(dy, dx)
            
            w_right = self.widths[i][0]
            w_left = self.widths[i][1]
            
            # Math to find wall points perpendicular to track
            # Right Wall
            rx = x + math.sin(yaw) * w_right
            ry = y - math.cos(yaw) * w_right
            
            # Left Wall
            lx = x - math.sin(yaw) * w_left
            ly = y + math.cos(yaw) * w_left
            
            # Add Vertical "Pillars" to see it better in 3D
            m_walls.points.append(Point(x=rx, y=ry, z=0.0))
            m_walls.points.append(Point(x=rx, y=ry, z=0.5))
            
            m_walls.points.append(Point(x=lx, y=ly, z=0.0))
            m_walls.points.append(Point(x=lx, y=ly, z=0.5))

        self.pub_walls.publish(m_walls)

def main(args=None):
    rclpy.init(args=args)
    node = TrackVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()