import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
import numpy as np
import casadi as ca
from scipy.spatial import KDTree
import math
import os

DT = 0.05
N_HORIZON = 20
L_VEHICLE = 0.33
W_TRACK = 4.0       
FILENAME = 'raceline.csv' 

# 1. TRACK MANAGER (PRESERVES ORIGINAL COORDINATES)
class TrackManager:
    def __init__(self, filename, width=4.0):
        self.width = width
        self.path = None
        
        if not os.path.exists(filename):
            filename = os.path.join(os.getcwd(), filename)

        if os.path.exists(filename):
            print(f"Loading track from {filename}...")
            try:
                data = np.loadtxt(filename, delimiter=',', skiprows=1)
                self.path = data[:, :2]
                # NOTE: NO OFFSETTING APPLIED HERE. RAW COORDINATES USED.
            except Exception as e:
                print(f"Error loading CSV: {e}")
                self.generate_fallback_track()
        else:
            print("CSV not found. Generating Ellipse.")
            self.generate_fallback_track()
            
        self.N_points = self.path.shape[0]
        self.kdtree = KDTree(self.path)
        self.process_path()

    def generate_fallback_track(self):
        theta = np.linspace(0, 2*np.pi, 200) 
        a, b = 10.0, 6.0 
        x = a * np.cos(theta)
        y = b * np.sin(theta)
        self.path = np.vstack((x, y)).T

    def process_path(self):
        x, y = self.path[:, 0], self.path[:, 1]
        dx, dy = np.gradient(x), np.gradient(y)
        ddx, ddy = np.gradient(dx), np.gradient(dy)
        self.headings = np.arctan2(dy, dx)
        denom = (dx**2 + dy**2)**1.5 + 1e-6
        self.curvature = (dx * ddy - dy * ddx) / denom

    def get_reference(self, x_car, y_car, horizon):
        dist, idx = self.kdtree.query([x_car, y_car])
        
        indices = np.arange(idx, idx + horizon + 1) % self.N_points
        ref_x = self.path[indices, 0]
        ref_y = self.path[indices, 1]
        ref_psi = np.unwrap(self.headings[indices])
        
        k_max = np.max(np.abs(self.curvature[indices]))
        v_limit = np.sqrt(4.0 / (k_max + 1e-3)) 
        v_limit = np.clip(v_limit, 1.0, 6.0) # Cap speed for stability
        
        return ref_x, ref_y, ref_psi, v_limit, idx, dist

# 2. LMPC LEARNER
class LMPCLearner:
    def __init__(self):
        self.X, self.U, self.X_next, self.X_nom = None, None, None, None
        self.tree = None
        self.Ae = np.zeros((4,4))
        self.Be = np.zeros((4,2))
        self.Ce = np.zeros((4,1))

    def add_data(self, X, U, X_next, X_nom):
        if self.X is None:
            self.X, self.U, self.X_next, self.X_nom = X, U, X_next, X_nom
        else:
            self.X = np.vstack((self.X, X))
            self.U = np.vstack((self.U, U))
            self.X_next = np.vstack((self.X_next, X_next))
            self.X_nom = np.vstack((self.X_nom, X_nom))
        self.tree = KDTree(self.X[:, :2])
        print(f"LMPC Memory Updated: {self.X.shape[0]} points")

    def update_error_model(self, x_curr):
        if self.X is None: return
        dist, idx = self.tree.query(x_curr[:2], k=15)
        w = np.where(dist/3.0 < 1, 0.75*(1-(dist/3.0)**2), 0.0)
        if np.sum(w) < 1e-3: return
        W = np.diag(w)
        Z = np.hstack((self.X[idx], self.U[idx], np.ones((len(idx), 1))))
        E = self.X_next[idx] - self.X_nom[idx]
        try:
            Theta = (np.linalg.inv(Z.T @ W @ Z + 1e-4*np.eye(Z.shape[1])) @ Z.T @ W @ E).T
            alpha = 0.1
            self.Ae = (1-alpha)*self.Ae + alpha*np.clip(Theta[:, 0:4], -2, 2)
            self.Be = (1-alpha)*self.Be + alpha*np.clip(Theta[:, 4:6], -2, 2)
            self.Ce = (1-alpha)*self.Ce + alpha*np.clip(Theta[:, 6:7], -1, 1)
        except: pass

# 3. CASADI MPC (With Damping)
class CasadiMPC:
    def __init__(self):
        self.opti = ca.Opti()
        self.N = N_HORIZON
        self.X = self.opti.variable(4, self.N + 1)
        self.U = self.opti.variable(2, self.N)
        self.P_x0 = self.opti.parameter(4)
        self.P_ref = self.opti.parameter(3, self.N + 1)
        self.P_v_tgt = self.opti.parameter(1)
        self.P_Ae = self.opti.parameter(4, 4)
        self.P_Be = self.opti.parameter(4, 2)
        self.P_Ce = self.opti.parameter(4, 1)
        self.P_u_prev = self.opti.parameter(2)

        cost = 0
        for k in range(self.N):
            err_path = 2.0*((self.X[0, k] - self.P_ref[0, k])**2 + (self.X[1, k] - self.P_ref[1, k])**2)
            err_vel = 1.0*(self.X[2, k] - self.P_v_tgt)**2
            
            # Slew Rate Damping (Essential for preventing oscillations)
            if k == 0:
                delta_u_acc = (self.U[0, k] - self.P_u_prev[0])**2
                delta_u_steer = (self.U[1, k] - self.P_u_prev[1])**2
            else:
                delta_u_acc = (self.U[0, k] - self.U[0, k-1])**2
                delta_u_steer = (self.U[1, k] - self.U[1, k-1])**2
            
            damping = 5.0 * delta_u_acc + 200.0 * delta_u_steer
            cost += err_path + err_vel + 0.1*self.U[0, k]**2 + damping
            
        self.opti.minimize(cost)

        for k in range(self.N):
            x_k = self.X[:, k]
            u_k = self.U[:, k]
            beta = ca.atan(0.5 * ca.tan(u_k[1]))
            x_nom = ca.vertcat(
                x_k[0] + x_k[2] * ca.cos(x_k[3] + beta) * DT,
                x_k[1] + x_k[2] * ca.sin(x_k[3] + beta) * DT,
                x_k[2] + u_k[0] * DT,
                x_k[3] + (x_k[2] / L_VEHICLE) * ca.sin(beta) * DT
            )
            x_next = x_nom + ca.mtimes(self.P_Ae, x_k) + ca.mtimes(self.P_Be, u_k) + self.P_Ce
            self.opti.subject_to(self.X[:, k+1] == x_next)

        self.opti.subject_to(self.X[:, 0] == self.P_x0)
        self.opti.subject_to(self.U[0, :] <= 5.0); self.opti.subject_to(self.U[0, :] >= -5.0)
        self.opti.subject_to(self.U[1, :] <= 0.45); self.opti.subject_to(self.U[1, :] >= -0.45)
        self.opti.subject_to(self.X[2, :] >= -1.0) 

        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.max_iter': 40, 'ipopt.warm_start_init_point': 'yes'}
        self.opti.solver('ipopt', opts)

    def solve(self, x0, ref_traj, v_target, Ae, Be, Ce, u_prev):
        self.opti.set_value(self.P_x0, x0)
        self.opti.set_value(self.P_ref, np.vstack(ref_traj))
        self.opti.set_value(self.P_v_tgt, v_target)
        self.opti.set_value(self.P_Ae, Ae)
        self.opti.set_value(self.P_Be, Be)
        self.opti.set_value(self.P_Ce, Ce)
        self.opti.set_value(self.P_u_prev, u_prev)
        self.opti.set_initial(self.X, np.repeat(x0.reshape(-1,1), self.N+1, axis=1))
        try:
            sol = self.opti.solve()
            return sol.value(self.U)[:, 0], sol.value(self.X)
        except:
            return np.array([-1.0, 0.0]), np.repeat(x0.reshape(-1,1), self.N+1, axis=1)

# 4. ROS NODE
class LMPCNode(Node):
    def __init__(self):
        super().__init__('lmpc_node')
        self.declare_parameter('track_file', 'raceline.csv')
        self.declare_parameter('drive_topic', '/drive')
        self.declare_parameter('odom_topic', '/odom')

        self.track = TrackManager(self.get_parameter('track_file').value, W_TRACK)
        self.mpc = CasadiMPC()
        self.learner = LMPCLearner()

        self.current_state = None 
        self.prev_state = None
        self.prev_u = np.zeros(2)
        
        # Yaw Unwrapping vars
        self.last_yaw = 0.0
        self.unwrapped_yaw = 0.0
        self.first_run = True

        self.lap_data = {'X':[], 'U':[], 'X_n':[], 'X_nom':[]}
        self.lap_count = 0
        self.aggressiveness = 0.5
        self.is_racing = False
        
        self.pub_drive = self.create_publisher(AckermannDriveStamped, self.get_parameter('drive_topic').value, 10)
        self.create_subscription(Odometry, self.get_parameter('odom_topic').value, self.odom_callback, 10)
        
        self.pub_viz_path = self.create_publisher(Marker, '/mpc_path', 10)
        self.pub_ref_path = self.create_publisher(Marker, '/reference_path', 10)
        self.pub_global_track = self.create_publisher(Marker, '/global_track', 10)
        self.pub_history = self.create_publisher(Marker, '/history_path', 10)
        self.pub_lap_text = self.create_publisher(Marker, '/lap_info', 10)

        self.create_timer(DT, self.control_loop)
        self.get_logger().info("LMPC Node Initialized. Waiting for Odom...")

    def odom_callback(self, msg):
        px, py = msg.pose.pose.position.x, msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        raw_yaw = math.atan2(siny_cosp, cosy_cosp)
        
        # --- YAW UNWRAPPING (Crucial for loop stability) ---
        if self.first_run:
            self.unwrapped_yaw = raw_yaw
            self.last_yaw = raw_yaw
            self.first_run = False
        else:
            diff = raw_yaw - self.last_yaw
            if diff > math.pi: diff -= 2*math.pi
            elif diff < -math.pi: diff += 2*math.pi
            self.unwrapped_yaw += diff
            self.last_yaw = raw_yaw

        v = np.sqrt(msg.twist.twist.linear.x**2 + msg.twist.twist.linear.y**2)
        if msg.twist.twist.linear.x < 0: v = -v
        
        self.current_state = np.array([px, py, v, self.unwrapped_yaw])

    def get_nominal_dynamics(self, x, u):
        x_next = np.zeros(4)
        beta = np.arctan(0.5 * np.tan(u[1]))
        x_next[0] = x[0] + x[2] * np.cos(x[3] + beta) * DT
        x_next[1] = x[1] + x[2] * np.sin(x[3] + beta) * DT
        x_next[2] = x[2] + u[0] * DT
        x_next[3] = x[3] + (x[2] / L_VEHICLE) * np.sin(beta) * DT
        return x_next

    def control_loop(self):
        if self.current_state is None: return
        x = self.current_state
        
        # --- SAFETY LOCK: Distance Check ---
        rx, ry, rpsi, v_max, idx, dist_to_track = self.track.get_reference(x[0], x[1], N_HORIZON)
        
        # If car is too far from track, DO NOT ENGAGE MPC
        if dist_to_track > 3.0:
            self.get_logger().warn(f"Car is {dist_to_track:.2f}m from track. Move closer to start!", throttle_duration_sec=1.0)
            
            # Send Stop Command
            msg = AckermannDriveStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "base_link"
            self.pub_drive.publish(msg)
            
            # Visualize where track is
            self.publish_global_track()
            return
        # -----------------------------------

        if self.prev_state is not None and self.is_racing:
            x_nom = self.get_nominal_dynamics(self.prev_state, self.prev_u)
            self.lap_data['X'].append(self.prev_state); self.lap_data['U'].append(self.prev_u)
            self.lap_data['X_n'].append(x); self.lap_data['X_nom'].append(x_nom)
            if len(self.lap_data['X']) % 10 == 0: self.learner.update_error_model(x)

        # Lap Logic
        if not self.is_racing and idx > 50: self.is_racing = True
        if self.is_racing and idx < 20 and len(self.lap_data['X']) > 200: self.finish_lap()

        v_tgt = v_max * self.aggressiveness
        u_opt, x_pred_traj = self.mpc.solve(x, [rx, ry, rpsi], v_tgt, self.learner.Ae, self.learner.Be, self.learner.Ce, self.prev_u)

        msg = AckermannDriveStamped()
        msg.drive.steering_angle = float(u_opt[1])
        msg.drive.acceleration = float(u_opt[0])
        msg.drive.speed = x[2] + u_opt[0]*DT
        self.pub_drive.publish(msg)

        self.publish_visualization(x_pred_traj, rx, ry)
        self.publish_global_track()
        self.publish_history()
        self.publish_lap_text()

        self.prev_state = x.copy(); self.prev_u = u_opt.copy()

    def finish_lap(self):
        self.get_logger().info(f"Lap {self.lap_count} Complete!")
        X_arr = np.array(self.lap_data['X'])
        if len(X_arr) > 50:
            self.learner.add_data(X_arr, np.array(self.lap_data['U']), np.array(self.lap_data['X_n']), np.array(self.lap_data['X_nom']))
            self.lap_count += 1
            self.aggressiveness = min(0.9, self.aggressiveness + 0.1)
        self.lap_data = {'X':[], 'U':[], 'X_n':[], 'X_nom':[]}
        self.is_racing = True 

    def publish_history(self):
        if len(self.lap_data['X']) < 2: return
        msg = Marker(); msg.header.frame_id = "map"; msg.id = 500; msg.type = Marker.LINE_STRIP; msg.action = Marker.ADD
        msg.scale.x = 0.08; msg.color.a = 1.0; msg.color.r = 1.0; msg.color.g = 0.0; msg.color.b = 1.0 
        for state in self.lap_data['X']: msg.points.append(Point(x=float(state[0]), y=float(state[1])))
        self.pub_history.publish(msg)

    def publish_lap_text(self):
        msg = Marker(); msg.header.frame_id = "map"; msg.id = 501; msg.type = Marker.TEXT_VIEW_FACING; msg.action = Marker.ADD
        msg.scale.z = 1.5; msg.color.a = 1.0; msg.color.r = 1.0; msg.color.g = 1.0; msg.color.b = 1.0 
        msg.pose.position.x = self.current_state[0]; msg.pose.position.y = self.current_state[1]; msg.pose.position.z = 1.0 
        msg.text = f"LAP: {self.lap_count}"
        self.pub_lap_text.publish(msg)

    def publish_global_track(self):
        msg = Marker(); msg.header.frame_id = "map"; msg.id = 1000; msg.type = Marker.LINE_STRIP; msg.action = Marker.ADD
        msg.scale.x = 0.1; msg.color.a = 1.0; msg.color.b = 1.0 
        for i in range(len(self.track.path)): msg.points.append(Point(x=float(self.track.path[i,0]), y=float(self.track.path[i,1])))
        msg.points.append(Point(x=float(self.track.path[0,0]), y=float(self.track.path[0,1])))
        self.pub_global_track.publish(msg)

    def publish_visualization(self, pred, rx, ry):
        msg = Marker(); msg.header.frame_id = "map"; msg.type = Marker.LINE_STRIP; msg.action = Marker.ADD
        msg.scale.x = 0.1; msg.color.a = 1.0; msg.color.g = 1.0
        for i in range(pred.shape[1]): msg.points.append(Point(x=float(pred[0,i]), y=float(pred[1,i])))
        self.pub_viz_path.publish(msg)
        ref = Marker(); ref.header.frame_id = "map"; ref.type = Marker.POINTS; ref.action = Marker.ADD
        ref.scale.x = 0.2; ref.scale.y = 0.2; ref.color.a = 1.0; ref.color.r = 1.0
        for i in range(len(rx)): ref.points.append(Point(x=float(rx[i]), y=float(ry[i])))
        self.pub_ref_path.publish(ref)

def main(args=None):
    rclpy.init(args=args)
    node = LMPCNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()