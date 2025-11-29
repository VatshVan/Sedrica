import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, PoseStamped
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker
import numpy as np
import casadi as ca
from scipy.spatial import KDTree
import math
import os

# --- CONSTANTS ---
DT = 0.05
N_HORIZON = 20
L_VEHICLE = 0.33
W_TRACK = 4.0       
FILENAME = 'raceline.csv' 

MAX_STEER_RAD = 0.52 
MAX_ACCEL = 5.0        

class TrackManager:
    def __init__(self, filename, width=4.0):
        self.width = width
        self.path = None
        if not os.path.exists(filename): filename = os.path.join(os.getcwd(), filename)
        if os.path.exists(filename):
            print(f"Loading track from {filename}...")
            try:
                data = np.loadtxt(filename, delimiter=',', skiprows=1)
                self.path = data[:, :2]
            except: self.generate_fallback_track()
        else: self.generate_fallback_track()
        self.N_points = self.path.shape[0]
        self.kdtree = KDTree(self.path)
        self.process_path()

    def generate_fallback_track(self):
        theta = np.linspace(0, 2*np.pi, 200); a, b = 10.0, 6.0 
        self.path = np.vstack((a * np.cos(theta), b * np.sin(theta))).T

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
        k_max = np.max(np.abs(self.curvature[indices]))
        v_limit = np.clip(np.sqrt(4.0 / (k_max + 1e-3)), 1.0, 4.0)
        return self.path[indices, 0], self.path[indices, 1], np.unwrap(self.headings[indices]), v_limit, idx, dist

class LMPCLearner:
    def __init__(self):
        self.X, self.U, self.X_next, self.X_nom = None, None, None, None
        self.tree = None; self.Ae = np.zeros((4,4)); self.Be = np.zeros((4,2)); self.Ce = np.zeros((4,1))

    def add_data(self, X, U, X_next, X_nom):
        if self.X is None: self.X, self.U, self.X_next, self.X_nom = X, U, X_next, X_nom
        else:
            self.X = np.vstack((self.X, X)); self.U = np.vstack((self.U, U))
            self.X_next = np.vstack((self.X_next, X_next)); self.X_nom = np.vstack((self.X_nom, X_nom))
        self.tree = KDTree(self.X[:, :2])

    def update_error_model(self, x_curr):
        if self.X is None: return
        dist, idx = self.tree.query(x_curr[:2], k=15)
        w = np.where(dist/3.0 < 1, 0.75*(1-(dist/3.0)**2), 0.0)
        if np.sum(w) < 1e-3: return
        Z = np.hstack((self.X[idx], self.U[idx], np.ones((len(idx), 1))))
        E = self.X_next[idx] - self.X_nom[idx]
        try:
            Theta = (np.linalg.inv(Z.T @ np.diag(w) @ Z + 1e-4*np.eye(Z.shape[1])) @ Z.T @ np.diag(w) @ E).T
            self.Ae = 0.9*self.Ae + 0.1*np.clip(Theta[:, 0:4], -2, 2)
            self.Be = 0.9*self.Be + 0.1*np.clip(Theta[:, 4:6], -2, 2)
            self.Ce = 0.9*self.Ce + 0.1*np.clip(Theta[:, 6:7], -1, 1)
        except: pass

class CasadiMPC:
    def __init__(self):
        self.opti = ca.Opti(); self.N = N_HORIZON
        self.X = self.opti.variable(4, self.N + 1); self.U = self.opti.variable(2, self.N)
        self.P_x0 = self.opti.parameter(4); self.P_ref = self.opti.parameter(3, self.N + 1)
        self.P_v_tgt = self.opti.parameter(1); self.P_u_prev = self.opti.parameter(2)
        self.P_Ae = self.opti.parameter(4, 4); self.P_Be = self.opti.parameter(4, 2); self.P_Ce = self.opti.parameter(4, 1)

        cost = 0
        for k in range(self.N):
            cost += 2.0*((self.X[0,k]-self.P_ref[0,k])**2 + (self.X[1,k]-self.P_ref[1,k])**2) + (self.X[2,k]-self.P_v_tgt)**2
            delta_u = (self.U[:, k] - (self.P_u_prev if k==0 else self.U[:, k-1]))**2
            cost += 5.0*delta_u[0] + 500.0*delta_u[1]
        self.opti.minimize(cost)

        for k in range(self.N):
            x_k, u_k = self.X[:, k], self.U[:, k]
            beta = ca.atan(0.5 * ca.tan(u_k[1]))
            x_nom = ca.vertcat(x_k[0] + x_k[2] * ca.cos(x_k[3] + beta) * DT,
                               x_k[1] + x_k[2] * ca.sin(x_k[3] + beta) * DT,
                               x_k[2] + u_k[0] * DT,
                               x_k[3] + (x_k[2] / L_VEHICLE) * ca.sin(beta) * DT)
            self.opti.subject_to(self.X[:, k+1] == x_nom + ca.mtimes(self.P_Ae, x_k) + ca.mtimes(self.P_Be, u_k) + self.P_Ce)

        self.opti.subject_to(self.X[:, 0] == self.P_x0)
        self.opti.subject_to(self.opti.bounded(-MAX_ACCEL, self.U[0, :], MAX_ACCEL))
        self.opti.subject_to(self.opti.bounded(-0.45, self.U[1, :], 0.45))
        self.opti.solver('ipopt', {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.warm_start_init_point': 'yes'})

    def solve(self, x0, ref, v_tgt, Ae, Be, Ce, u_prev):
        self.opti.set_value(self.P_x0, x0); self.opti.set_value(self.P_ref, np.vstack(ref))
        self.opti.set_value(self.P_v_tgt, v_tgt); self.opti.set_value(self.P_Ae, Ae)
        self.opti.set_value(self.P_Be, Be); self.opti.set_value(self.P_Ce, Ce); self.opti.set_value(self.P_u_prev, u_prev)
        self.opti.set_initial(self.X, np.repeat(x0.reshape(-1,1), self.N+1, axis=1))
        try: sol = self.opti.solve(); return sol.value(self.U)[:, 0], sol.value(self.X)
        except: return np.array([-1.0, 0.0]), np.repeat(x0.reshape(-1,1), self.N+1, axis=1)

class LMPCNode(Node):
    def __init__(self):
        super().__init__('lmpc_node')
        
        # --- FIXED: Use correct command topics and IPS topic ---
        self.declare_parameter('throttle_topic', '/autodrive/f1tenth_1/throttle_command')
        self.declare_parameter('steering_topic', '/autodrive/f1tenth_1/steering_command')
        self.declare_parameter('pose_topic', '/autodrive/f1tenth_1/ips')

        self.track = TrackManager('raceline.csv', W_TRACK)
        self.mpc = CasadiMPC()
        self.learner = LMPCLearner()

        self.current_state = None; self.prev_state = None; self.prev_u = np.zeros(2)
        self.prev_pose = None; self.prev_pose_time = None
        self.last_yaw = 0.0; self.unwrapped_yaw = 0.0; self.first_run = True
        self.lap_data = {'X':[], 'U':[], 'X_n':[], 'X_nom':[]}; self.lap_count = 0; self.aggressiveness = 0.5; self.is_racing = False

        self.pub_throttle = self.create_publisher(Float32, self.get_parameter('throttle_topic').value, 10)
        self.pub_steering = self.create_publisher(Float32, self.get_parameter('steering_topic').value, 10)
        self.pub_viz_path = self.create_publisher(Marker, '/mpc_path', 10)
        self.pub_ref_path = self.create_publisher(Marker, '/reference_path', 10)
        self.pub_global_track = self.create_publisher(Marker, '/global_track', 10)

        # --- IMPORTANT CHANGE: SUBSCRIBE TO POINT (Not PoseStamped) ---
        self.create_subscription(Point, self.get_parameter('pose_topic').value, self._pose_cb, 10)
        
        self.create_timer(DT, self.control_loop)
        self.get_logger().info("LMPC Node Waiting for Point Data...")

    def _pose_cb(self, msg):
        # --- FIXED: Read directly from Point msg (x, y) ---
        px = msg.x
        py = msg.y
        
        # --- FIXED: Calculate Yaw from Motion ---
        raw_yaw = self.last_yaw
        if self.prev_pose is not None:
             dx = px - self.prev_pose[0]
             dy = py - self.prev_pose[1]
             # Only update yaw if we moved enough to be sure (reduce noise)
             if math.hypot(dx, dy) > 0.001:
                 raw_yaw = math.atan2(dy, dx)
        
        # Unwrap yaw to prevent jumps from PI to -PI
        if self.first_run:
            self.unwrapped_yaw = raw_yaw; self.last_yaw = raw_yaw; self.first_run = False
        else:
            diff = raw_yaw - self.last_yaw
            if diff > math.pi: diff -= 2*math.pi
            elif diff < -math.pi: diff += 2*math.pi
            self.unwrapped_yaw += diff; self.last_yaw = raw_yaw

        t = self.get_clock().now().nanoseconds / 1e9
        v = 0.0
        if self.prev_pose_time is not None:
            dt = t - self.prev_pose_time
            if dt > 1e-4:
                dist = math.hypot(px - self.prev_pose[0], py - self.prev_pose[1])
                v = dist / dt
                # Check if reversing (based on yaw projection)
                if (px - self.prev_pose[0])*math.cos(self.unwrapped_yaw) + (py - self.prev_pose[1])*math.sin(self.unwrapped_yaw) < 0: v = -v

        self.prev_pose = (px, py); self.prev_pose_time = t
        self.current_state = np.array([px, py, v, self.unwrapped_yaw])

    def control_loop(self):
        if self.current_state is None: return
        x = self.current_state
        rx, ry, rpsi, v_max, idx, dist = self.track.get_reference(x[0], x[1], N_HORIZON)
        
        if self.prev_state is not None and self.is_racing:
             self.lap_data['X'].append(self.prev_state); self.lap_data['U'].append(self.prev_u)
             self.lap_data['X_n'].append(x); self.lap_data['X_nom'].append(self.get_nominal_dynamics(self.prev_state, self.prev_u))
             if len(self.lap_data['X']) % 10 == 0: self.learner.update_error_model(x)

        if not self.is_racing and idx > 50: self.is_racing = True; self.get_logger().info("Race Started!")
        if self.is_racing and idx < 20 and len(self.lap_data['X']) > 200: self.finish_lap()

        u_opt, pred = self.mpc.solve(x, [rx, ry, rpsi], v_max * self.aggressiveness, self.learner.Ae, self.learner.Be, self.learner.Ce, self.prev_u)
        
        t_msg = Float32(); t_msg.data = float(np.clip(u_opt[0]/MAX_ACCEL, -1.0, 1.0))
        s_msg = Float32(); s_msg.data = float(np.clip(u_opt[1]/MAX_STEER_RAD, -1.0, 1.0))
        self.pub_throttle.publish(t_msg); self.pub_steering.publish(s_msg)
        
        m = Marker(); m.header.frame_id = "map"; m.type = Marker.LINE_STRIP; m.action = Marker.ADD; m.scale.x = 0.1; m.color.a = 1.0; m.color.g = 1.0
        for i in range(pred.shape[1]): m.points.append(Point(x=float(pred[0,i]), y=float(pred[1,i])))
        self.pub_viz_path.publish(m); self.publish_global_track()
        self.prev_state = x.copy(); self.prev_u = u_opt.copy()

    def get_nominal_dynamics(self, x, u):
        beta = np.arctan(0.5 * np.tan(u[1]))
        return np.array([x[0] + x[2] * np.cos(x[3] + beta) * DT, x[1] + x[2] * np.sin(x[3] + beta) * DT, x[2] + u[0] * DT, x[3] + (x[2] / L_VEHICLE) * np.sin(beta) * DT])

    def finish_lap(self):
        self.learner.add_data(np.array(self.lap_data['X']), np.array(self.lap_data['U']), np.array(self.lap_data['X_n']), np.array(self.lap_data['X_nom']))
        self.lap_count += 1; self.aggressiveness = min(0.9, self.aggressiveness + 0.1)
        self.lap_data = {'X':[], 'U':[], 'X_n':[], 'X_nom':[]}; self.get_logger().info(f"Lap {self.lap_count} Done!")

    def publish_global_track(self):
        msg = Marker(); msg.header.frame_id = "map"; msg.id = 1000; msg.type = Marker.LINE_STRIP; msg.action = Marker.ADD; msg.scale.x = 0.1; msg.color.a = 1.0; msg.color.b = 1.0 
        for i in range(len(self.track.path)): msg.points.append(Point(x=float(self.track.path[i,0]), y=float(self.track.path[i,1])))
        msg.points.append(Point(x=float(self.track.path[0,0]), y=float(self.track.path[0,1])))
        self.pub_global_track.publish(msg)

def main(args=None):
    rclpy.init(args=args); node = LMPCNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()