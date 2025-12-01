import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, PoseWithCovarianceStamped
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import casadi as ca
from scipy.spatial import KDTree
import math
import pandas as pd
import os

# --- CONSTANTS ---
DT = 0.05
N_HORIZON = 20
L_VEHICLE = 0.33
W_TRACK = 4.0
FILENAME = 'raceline_2.csv' 

MAX_STEER_RAD = 0.52 
MAX_ACCEL = 5.0 
SAFETY_MARGIN = 0.3 

# --- LAP 0 SETTINGS ---
LAP0_SPEED = 2.0 

class TrackManager:
    def __init__(self, filename, width=4.0):
        self.width = width
        self.path = None
        self.widths = None
        
        if not os.path.exists(filename): filename = os.path.join(os.getcwd(), filename)
        if os.path.exists(filename):
            print(f"Loading track from {filename}...")
            try:
                data = np.loadtxt(filename, delimiter=',', skiprows=1)
            except:
                try: data = np.loadtxt(filename, skiprows=1)
                except: self.generate_fallback_track(); return

            if data.shape[1] == 3:
                self.path = data[:, :2]
                w = data[:, 2].reshape(-1, 1)
                self.widths = np.hstack((w, w)) 
            elif data.shape[1] >= 4:
                self.path = data[:, :2]
                self.widths = data[:, 2:4] 
            else:
                self.path = data[:, :2]
                self.widths = np.ones((data.shape[0], 2)) * (width / 2.0)
        else:
            self.generate_fallback_track()
            
        self.N_points = self.path.shape[0]
        self.kdtree = KDTree(self.path)
        self.process_path()

    def generate_fallback_track(self):
        theta = np.linspace(0, 2*np.pi, 200); a, b = 10.0, 6.0 
        self.path = np.vstack((a * np.cos(theta), b * np.sin(theta))).T
        self.widths = np.ones((200, 2)) * 2.0

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
        ref_x = self.path[indices, 0]; ref_y = self.path[indices, 1]
        ref_psi = np.unwrap(self.headings[indices])
        ref_w_right = self.widths[indices, 0]; ref_w_left = self.widths[indices, 1]
        k_max = np.max(np.abs(self.curvature[indices]))
        v_limit = np.clip(np.sqrt(4.0 / (k_max + 1e-3)), 1.0, 4.0)
        return ref_x, ref_y, ref_psi, ref_w_right, ref_w_left, v_limit, idx, dist

class LMPCLearner:
    def __init__(self):
        self.X_log = []
        self.U_log = []
        self.X_next_log = []
        
        # We only learn the Bias (Ce) to ensure stability
        # Ae and Be are 0 (Trust the Nominal Model gradients)
        self.Ce = np.zeros((4,1))
        self.tree = None

    def add_data(self, X, U, X_next):
        if len(self.X_log) == 0:
            self.X_log = X; self.U_log = U; self.X_next_log = X_next
        else:
            self.X_log = np.vstack((self.X_log, X))
            self.U_log = np.vstack((self.U_log, U))
            self.X_next_log = np.vstack((self.X_next_log, X_next))
        
        self.tree = KDTree(self.X_log[:, :2])
        print(f"LMPC Memory: {self.X_log.shape[0]} points. Learning Updated.")

    def get_nominal_pred(self, x, u):
        # Kinematic Bicycle Model (Same as MPC)
        beta = math.atan(0.5 * math.tan(u[1]))
        x_next = np.zeros(4)
        x_next[0] = x[0] + x[2] * math.cos(x[3] + beta) * DT
        x_next[1] = x[1] + x[2] * math.sin(x[3] + beta) * DT
        x_next[2] = x[2] + u[0] * DT
        x_next[3] = x[3] + (x[2] / L_VEHICLE) * math.sin(beta) * DT
        return x_next

    def update_error_model(self, x_curr):
        if self.tree is None: return
        
        # Find 20 nearest points in history
        dist, idx = self.tree.query(x_curr[:2], k=20)
        
        error_sum = np.zeros(4)
        valid_points = 0
        
        for i in idx:
            # Reconstruct what the Nominal Model WOULD have predicted
            x_hist = self.X_log[i]
            u_hist = self.U_log[i]
            x_next_actual = self.X_next_log[i]
            
            x_next_nominal = self.get_nominal_pred(x_hist, u_hist)
            
            # Error = Real - Nominal
            error_sum += (x_next_actual - x_next_nominal)
            valid_points += 1
            
        if valid_points > 0:
            # Average error becomes our "Ce" (Additive Disturbance)
            # We apply a low-pass filter (alpha=0.2) to smooth it out
            new_Ce = (error_sum / valid_points).reshape(4,1)
            self.Ce = 0.8 * self.Ce + 0.2 * new_Ce

class CasadiMPC:
    def __init__(self):
        self.opti = ca.Opti(); self.N = N_HORIZON
        self.X = self.opti.variable(4, self.N + 1); self.U = self.opti.variable(2, self.N)
        self.P_x0 = self.opti.parameter(4); self.P_ref = self.opti.parameter(3, self.N + 1)
        self.P_v_tgt = self.opti.parameter(1); self.P_u_prev = self.opti.parameter(2)
        self.P_w_right = self.opti.parameter(self.N + 1); self.P_w_left = self.opti.parameter(self.N + 1)
        
        # Learned Error Parameter (Additive Only)
        self.P_Ce = self.opti.parameter(4, 1)
        
        self.Sl_right = self.opti.variable(self.N + 1); self.Sl_left = self.opti.variable(self.N + 1)

        cost = 0
        for k in range(self.N):
            # Tracking Cost
            cost += 2.0*((self.X[0,k]-self.P_ref[0,k])**2 + (self.X[1,k]-self.P_ref[1,k])**2) 
            cost += 1.0*(self.X[2,k]-self.P_v_tgt)**2
            
            # Input Smoothness
            if k == 0:
                delta_u_acc = (self.U[0, k] - self.P_u_prev[0])**2
                delta_u_steer = (self.U[1, k] - self.P_u_prev[1])**2
            else:
                delta_u_acc = (self.U[0, k] - self.U[0, k-1])**2
                delta_u_steer = (self.U[1, k] - self.U[1, k-1])**2
            
            cost += 5.0*delta_u_acc + 50.0*delta_u_steer + 0.1*self.U[0, k]**2
            cost += 100000.0 * (self.Sl_right[k]**2 + self.Sl_left[k]**2)

        self.opti.minimize(cost)

        for k in range(self.N):
            x_k, u_k = self.X[:, k], self.U[:, k]
            
            # Track Constraints
            ref_x, ref_y, ref_psi = self.P_ref[0, k], self.P_ref[1, k], self.P_ref[2, k]
            lateral_error = -ca.sin(ref_psi) * (x_k[0] - ref_x) + ca.cos(ref_psi) * (x_k[1] - ref_y)
            max_right = self.P_w_right[k] - SAFETY_MARGIN; max_left = self.P_w_left[k] - SAFETY_MARGIN
            self.opti.subject_to(lateral_error <= max_right + self.Sl_right[k])
            self.opti.subject_to(lateral_error >= -max_left - self.Sl_left[k])
            self.opti.subject_to(self.Sl_right[k] >= 0); self.opti.subject_to(self.Sl_left[k] >= 0)

            # Dynamics: x_next = f(x,u) + Ce
            beta = ca.atan(0.5 * ca.tan(u_k[1]))
            x_nom = ca.vertcat(x_k[0] + x_k[2] * ca.cos(x_k[3] + beta) * DT,
                               x_k[1] + x_k[2] * ca.sin(x_k[3] + beta) * DT,
                               x_k[2] + u_k[0] * DT,
                               x_k[3] + (x_k[2] / L_VEHICLE) * ca.sin(beta) * DT)
            
            # Add the learned error term here
            self.opti.subject_to(self.X[:, k+1] == x_nom + self.P_Ce)

        self.opti.subject_to(self.X[:, 0] == self.P_x0)
        self.opti.subject_to(self.U[0, :] <= MAX_ACCEL); self.opti.subject_to(self.U[0, :] >= -MAX_ACCEL)
        self.opti.subject_to(self.U[1, :] <= 0.45); self.opti.subject_to(self.U[1, :] >= -0.45)
        self.opti.solver('ipopt', {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.max_iter': 80, 'ipopt.warm_start_init_point': 'yes'})

    def solve(self, x0, ref, w_right, w_left, v_tgt, Ce, u_prev):
        self.opti.set_value(self.P_x0, x0); self.opti.set_value(self.P_ref, np.vstack(ref))
        self.opti.set_value(self.P_w_right, w_right); self.opti.set_value(self.P_w_left, w_left)
        self.opti.set_value(self.P_v_tgt, v_tgt)
        self.opti.set_value(self.P_Ce, Ce) # Pass the learned error
        self.opti.set_value(self.P_u_prev, u_prev)
        
        self.opti.set_initial(self.X, np.repeat(x0.reshape(-1,1), self.N+1, axis=1))
        self.opti.set_initial(self.Sl_right, 0.0); self.opti.set_initial(self.Sl_left, 0.0)
        
        try: 
            sol = self.opti.solve()
            return sol.value(self.U)[:, 0], sol.value(self.X)
        except:
            return None, None

class LMPCNode(Node):
    def __init__(self):
        super().__init__('lmpc_node')
        self.declare_parameter('throttle_topic', '/autodrive/f1tenth_1/throttle_command')
        self.declare_parameter('steering_topic', '/autodrive/f1tenth_1/steering_command')
        self.declare_parameter('pose_topic', '/autodrive/f1tenth_1/ips')

        self.track = TrackManager(FILENAME, W_TRACK)
        self.mpc = CasadiMPC()
        self.learner = LMPCLearner()

        self.current_state = None; self.prev_state = None; self.prev_u = np.zeros(2)
        self.prev_pose = None; self.prev_pose_time = None
        self.last_yaw = 0.0; self.unwrapped_yaw = 0.0; self.first_run = True
        self.lap_data = {'X':[], 'U':[], 'X_next':[]}; self.lap_count = 0; self.aggressiveness = 0.5; self.is_racing = False

        self.pub_throttle = self.create_publisher(Float32, self.get_parameter('throttle_topic').value, 10)
        self.pub_steering = self.create_publisher(Float32, self.get_parameter('steering_topic').value, 10)
        self.pub_reset = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        
        self.pub_viz_path = self.create_publisher(Marker, '/mpc_path', 10)
        self.pub_ref_path = self.create_publisher(Marker, '/reference_path', 10)
        self.pub_goal = self.create_publisher(Marker, '/goal_point', 10)
        self.pub_global_track = self.create_publisher(Marker, '/global_track', 10)
        self.pub_boundaries = self.create_publisher(Marker, '/track_boundaries', 10)
        self.pub_lap_info = self.create_publisher(Marker, '/lap_info', 10)

        self.create_subscription(Point, self.get_parameter('pose_topic').value, self._pose_cb, 10)
        self.create_timer(DT, self.control_loop)
        self.get_logger().info("LMPC Node Initialized. Waiting for Data...")

    def _pose_cb(self, msg):
        px = msg.x; py = msg.y
        raw_yaw = self.last_yaw
        if self.prev_pose is not None:
             dx = px - self.prev_pose[0]; dy = py - self.prev_pose[1]
             if math.hypot(dx, dy) > 0.05: raw_yaw = math.atan2(dy, dx)
        
        if self.first_run: self.unwrapped_yaw = raw_yaw; self.last_yaw = raw_yaw; self.first_run = False
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
                if (px - self.prev_pose[0])*math.cos(self.unwrapped_yaw) + (py - self.prev_pose[1])*math.sin(self.unwrapped_yaw) < 0: v = -v

        self.prev_pose = (px, py); self.prev_pose_time = t
        self.current_state = np.array([px, py, v, self.unwrapped_yaw])

    # --- NEW FUNCTION: AUTO-RECOVERY using ALPP Logic ---
    def run_recovery_control(self, x, rx, ry):
        dx = rx - x[0]; dy = ry - x[1]
        dist_to_target = math.hypot(dx, dy)
        local_x = dx * math.cos(-x[3]) - dy * math.sin(-x[3])
        local_y = dx * math.sin(-x[3]) + dy * math.cos(-x[3])
        
        lookahead_dist = max(1.0, dist_to_target)
        alpha = math.atan2(local_y, local_x)
        steer_cmd = math.atan2(2.0 * L_VEHICLE * math.sin(alpha), lookahead_dist)
        
        target_v = 1.5 # Recovery Speed
        if local_x < 0: target_v = -1.5; steer_cmd = -steer_cmd # Reverse
            
        accel_cmd = (target_v - x[2]) 
        
        t_msg = Float32(); t_msg.data = float(np.clip(accel_cmd, -1.0, 1.0))
        s_msg = Float32(); s_msg.data = float(np.clip(steer_cmd, -MAX_STEER_RAD, MAX_STEER_RAD))
        self.pub_throttle.publish(t_msg); self.pub_steering.publish(s_msg)
        self.get_logger().warn(f"OFF TRACK! ALPP Recovery... Dist: {dist_to_target:.2f}m")
        self.prev_state = None; self.prev_u = np.zeros(2)

    def run_pure_pursuit(self, x, idx, v_max):
        lookahead_dist = 2.0; lookahead_idx = int(lookahead_dist / 0.1)
        target_i = (idx + lookahead_idx) % self.track.N_points
        goal_point = self.track.path[target_i]
        
        gm = Marker(); gm.header.frame_id = 'map'; gm.type = Marker.SPHERE; gm.action = Marker.ADD
        gm.pose.position.x = float(goal_point[0]); gm.pose.position.y = float(goal_point[1])
        gm.scale.x = 0.5; gm.scale.y = 0.5; gm.scale.z = 0.5; gm.color.a = 1.0; gm.color.r = 1.0
        self.pub_goal.publish(gm)

        dx = float(goal_point[0] - x[0]); dy = float(goal_point[1] - x[1])
        local_x = dx * math.cos(-x[3]) - dy * math.sin(-x[3])
        local_y = dx * math.sin(-x[3]) + dy * math.cos(-x[3])
        alpha = math.atan2(local_y, local_x)
        
        steer_cmd = math.atan2(2.0 * L_VEHICLE * math.sin(alpha), lookahead_dist)
        target_v = LAP0_SPEED if self.lap_count == 0 else v_max
        accel_cmd = (target_v - x[2])
        return np.array([accel_cmd, steer_cmd]), np.zeros((2, N_HORIZON))

    def control_loop(self):
        if self.current_state is None: return
        x = self.current_state
        rx, ry, rpsi, w_right, w_left, v_max, idx, dist = self.track.get_reference(x[0], x[1], N_HORIZON)
        
        # Crash Check
        current_dx = x[0] - rx[0]; current_dy = x[1] - ry[0]
        lat_error = -math.sin(rpsi[0]) * current_dx + math.cos(rpsi[0]) * current_dy
        if lat_error > w_right[0] + 0.1 or lat_error < -w_left[0] - 0.1:
            self.run_recovery_control(x, rx[0], ry[0]); return 

        if self.prev_state is not None and self.is_racing:
             self.lap_data['X'].append(self.prev_state); self.lap_data['U'].append(self.prev_u)
             self.lap_data['X_next'].append(x)
             # Update Learning (Additive Error Only)
             if self.lap_count > 0: self.learner.update_error_model(x)

        if not self.is_racing and idx > 50: self.is_racing = True; self.get_logger().info(f"Lap {self.lap_count} STARTED!")
        if self.is_racing and idx < 20 and len(self.lap_data['X']) > 200: self.finish_lap()

        self.get_logger().info(f"Speed: {x[2]:.2f}, Yaw: {x[3]:.2f}")

        if self.lap_count == 0:
            u_opt, pred = self.run_pure_pursuit(x, idx, v_max)
        else:
            # LAP 1+: Use LMPC with Additive Error Correction (Robust)
            u_opt, pred = self.mpc.solve(x, [rx, ry, rpsi], w_right, w_left, v_max * self.aggressiveness, 
                                         self.learner.Ce, self.prev_u)
            if u_opt is None:
                self.get_logger().warn("LMPC Solver Failed! Using PP Fallback.")
                u_opt, pred = self.run_pure_pursuit(x, idx, v_max)
        
        t_msg = Float32(); t_msg.data = float(np.clip(u_opt[0]/MAX_ACCEL, -1.0, 1.0))
        s_msg = Float32(); s_msg.data = float(np.clip(u_opt[1]/MAX_STEER_RAD, -1.0, 1.0))
        self.pub_throttle.publish(t_msg); self.pub_steering.publish(s_msg)
        
        # Viz
        m = Marker(); m.header.frame_id = "map"; m.type = Marker.LINE_STRIP; m.action = Marker.ADD; m.scale.x = 0.1; m.color.a = 1.0; m.color.g = 1.0
        for i in range(pred.shape[1]): m.points.append(Point(x=float(pred[0,i]), y=float(pred[1,i])))
        self.pub_viz_path.publish(m)
        self.publish_global_track(); self.publish_lap_info()
        self.prev_state = x.copy(); self.prev_u = u_opt.copy()

    def finish_lap(self):
        self.learner.add_data(np.array(self.lap_data['X']), np.array(self.lap_data['U']), np.array(self.lap_data['X_next']))
        self.get_logger().info(f"Lap {self.lap_count} COMPLETE.")
        self.lap_count += 1
        if self.lap_count > 0: self.aggressiveness = min(1.0, self.aggressiveness + 0.1)
        self.lap_data = {'X':[], 'U':[], 'X_next':[]}

    def publish_global_track(self):
        msg = Marker(); msg.header.frame_id = "map"; msg.id = 1000; msg.type = Marker.LINE_STRIP; msg.action = Marker.ADD; msg.scale.x = 0.1; msg.color.a = 1.0; msg.color.b = 1.0 
        for i in range(len(self.track.path)): msg.points.append(Point(x=float(self.track.path[i,0]), y=float(self.track.path[i,1])))
        self.pub_global_track.publish(msg)
        b_msg = Marker(); b_msg.header.frame_id = "map"; b_msg.id = 1001; b_msg.type = Marker.LINE_LIST; b_msg.action = Marker.ADD; b_msg.scale.x = 0.05; b_msg.color.a = 1.0; b_msg.color.r = 1.0
        for i in range(0, len(self.track.path), 5):
             x, y = self.track.path[i]; yaw = self.track.headings[i]; wr, wl = self.track.widths[i]
             lx = x - math.sin(yaw)*wl; ly = y + math.cos(yaw)*wl
             b_msg.points.append(Point(x=lx, y=ly, z=0.0)); b_msg.points.append(Point(x=lx, y=ly, z=0.5))
             rx = x + math.sin(yaw)*wr; ry = y - math.cos(yaw)*wr
             b_msg.points.append(Point(x=rx, y=ry, z=0.0)); b_msg.points.append(Point(x=rx, y=ry, z=0.5))
        self.pub_boundaries.publish(b_msg)

    def publish_lap_info(self):
        msg = Marker(); msg.header.frame_id = "map"; msg.id = 2000; msg.type = Marker.TEXT_VIEW_FACING; msg.action = Marker.ADD; msg.scale.z = 2.0; msg.color.a = 1.0; msg.color.r = 1.0; msg.color.g = 1.0
        if self.current_state is not None:
             msg.pose.position.x = self.current_state[0]; msg.pose.position.y = self.current_state[1]; msg.pose.position.z = 2.0
             msg.text = "MODE: ALPP (Lap 0)" if self.lap_count == 0 else f"MODE: LMPC (Lap {self.lap_count})"
        self.pub_lap_info.publish(msg)

def main(args=None):
    rclpy.init(args=args); node = LMPCNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()