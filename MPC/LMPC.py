import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, Point
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import casadi as ca
from scipy.spatial import KDTree
import math
import os
import sys

DT = 0.05
N_HORIZON = 20
L_VEHICLE = 0.33
W_TRACK = 4.0       
FILENAME = 'raceline.csv' 

# 1. TRACK MANAGER
class TrackManager:
    def __init__(self, filename, width=4.0):
        self.width = width
        self.path = None
        
        # Check if absolute path or local
        if not os.path.exists(filename):
            # Try looking in current directory if full path fails
            filename = os.path.join(os.getcwd(), filename)

        if os.path.exists(filename):
            print(f"Loading track from {filename}...")
            try:
                data = np.loadtxt(filename, delimiter=',', skiprows=1)
                self.path = data[:, :2]
            except Exception as e:
                print(f"Failed to load CSV: {e}")
                self.generate_fallback_track()
        else:
            print(f"raceline.csv not found at {filename}. Generating Ellipse.")
            self.generate_fallback_track()
            
        self.N_points = self.path.shape[0]
        self.kdtree = KDTree(self.path)
        self.process_path()

    def generate_fallback_track(self):
        theta = np.linspace(0, 2*np.pi, 1000)
        a, b = 10.0, 6.0 # Smaller for indoor testing, increase for sim
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
        
        # Boundaries for Viz (Optional)
        norm = np.sqrt(dx**2 + dy**2) + 1e-6
        nx, ny = -dy/norm, dx/norm
        self.inner = np.vstack((x + nx*self.width/2, y + ny*self.width/2)).T
        self.outer = np.vstack((x - nx*self.width/2, y - ny*self.width/2)).T

    def get_reference(self, x_car, y_car, horizon):
        _, idx = self.kdtree.query([x_car, y_car])
        
        # Cyclic buffer logic
        indices = np.arange(idx, idx + horizon + 1) % self.N_points
        
        ref_x = self.path[indices, 0]
        ref_y = self.path[indices, 1]
        # Unwrapping heading is crucial for the solver to not spin 360
        ref_psi = np.unwrap(self.headings[indices])
        
        k_max = np.max(np.abs(self.curvature[indices]))
        v_limit = np.sqrt(4.0 / (k_max + 1e-3)) # Slightly conservative friction
        v_limit = np.clip(v_limit, 1.0, 8.0) 
        
        return ref_x, ref_y, ref_psi, v_limit, idx

# ==============================================================================
# 2. LMPC LEARNER (Unchanged)
# ==============================================================================
class LMPCLearner:
    def __init__(self):
        self.X, self.U, self.X_next, self.X_nom = None, None, None, None
        self.tree = None
        self.Ae = np.zeros((4,4))
        self.Be = np.zeros((4,2))
        self.Ce = np.zeros((4,1))

    def add_data(self, X, U, X_next, X_nom):
        if self.X is None:
            self.X, self.U = X, U
            self.X_next, self.X_nom = X_next, X_nom
        else:
            self.X = np.vstack((self.X, X))
            self.U = np.vstack((self.U, U))
            self.X_next = np.vstack((self.X_next, X_next))
            self.X_nom = np.vstack((self.X_nom, X_nom))
        self.tree = KDTree(self.X[:, :2])
        print(f"LMPC Memory Updated. Total Data Points: {self.X.shape[0]}")

    def update_error_model(self, x_curr):
        if self.X is None: return
        
        dist, idx = self.tree.query(x_curr[:2], k=15)
        h = 3.0
        w = np.where(dist/h < 1, 0.75*(1-(dist/h)**2), 0.0)
        
        if np.sum(w) < 1e-3: return
        
        W = np.diag(w)
        X_loc = self.X[idx]; U_loc = self.U[idx]
        E_loc = self.X_next[idx] - self.X_nom[idx]
        
        ones = np.ones((len(idx), 1))
        Z = np.hstack((X_loc, U_loc, ones))
        
        try:
            reg = 1e-4 * np.eye(Z.shape[1])
            Theta = np.linalg.inv(Z.T @ W @ Z + reg) @ Z.T @ W @ E_loc
            Theta = Theta.T 
            
            # Inertia update
            alpha = 0.1
            self.Ae = (1-alpha)*self.Ae + alpha*np.clip(Theta[:, 0:4], -2, 2)
            self.Be = (1-alpha)*self.Be + alpha*np.clip(Theta[:, 4:6], -2, 2)
            self.Ce = (1-alpha)*self.Ce + alpha*np.clip(Theta[:, 6:7], -1, 1)
        except np.linalg.LinAlgError:
            pass

# ==============================================================================
# 3. CASADI MPC (MODIFIED FOR STABILITY)
# ==============================================================================
class CasadiMPC:
    def __init__(self):
        self.opti = ca.Opti()
        self.N = N_HORIZON
        
        self.X = self.opti.variable(4, self.N + 1) # x, y, v, psi
        self.U = self.opti.variable(2, self.N)     # acc, delta
        
        self.P_x0    = self.opti.parameter(4)
        self.P_ref   = self.opti.parameter(3, self.N + 1)
        self.P_v_tgt = self.opti.parameter(1)
        self.P_Ae    = self.opti.parameter(4, 4)
        self.P_Be    = self.opti.parameter(4, 2)
        self.P_Ce    = self.opti.parameter(4, 1)
        
        # --- NEW: Previous Control for Damping ---
        self.P_u_prev = self.opti.parameter(2)

        cost = 0
        for k in range(self.N):
            # --- MODIFIED: Reduced Path Weight (50.0 -> 2.0) ---
            err_path = 2.0*((self.X[0, k] - self.P_ref[0, k])**2 + \
                             (self.X[1, k] - self.P_ref[1, k])**2)
            
            err_vel = 1.0*(self.X[2, k] - self.P_v_tgt)**2
            reg_u = 0.1*self.U[0, k]**2 + 1.0*self.U[1, k]**2 
            
            # --- NEW: Slew Rate Cost (Damping) ---
            if k == 0:
                delta_u_acc = (self.U[0, k] - self.P_u_prev[0])**2
                delta_u_steer = (self.U[1, k] - self.P_u_prev[1])**2
            else:
                delta_u_acc = (self.U[0, k] - self.U[0, k-1])**2
                delta_u_steer = (self.U[1, k] - self.U[1, k-1])**2
                
            damping = 5.0 * delta_u_acc + 200.0 * delta_u_steer
            
            cost += err_path + err_vel + reg_u + damping
            
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
            x_next = x_nom + ca.mtimes(self.P_Ae, x_k) + \
                             ca.mtimes(self.P_Be, u_k) + \
                             self.P_Ce
            self.opti.subject_to(self.X[:, k+1] == x_next)

        self.opti.subject_to(self.X[:, 0] == self.P_x0)
        self.opti.subject_to(self.U[0, :] <= 5.0)
        self.opti.subject_to(self.U[0, :] >= -5.0)
        self.opti.subject_to(self.U[1, :] <= 0.45) # ~25 degrees
        self.opti.subject_to(self.U[1, :] >= -0.45)
        self.opti.subject_to(self.X[2, :] >= -1.0) # Allow slight reverse for recovery

        # IPOPT options for speed
        opts = {
            'ipopt.print_level': 0, 
            'print_time': 0, 
            'ipopt.max_iter': 40,
            'ipopt.warm_start_init_point': 'yes'
        }
        self.opti.solver('ipopt', opts)

    # --- MODIFIED: Added u_prev argument ---
    def solve(self, x0, ref_traj, v_target, Ae, Be, Ce, u_prev):
        self.opti.set_value(self.P_x0, x0)
        self.opti.set_value(self.P_ref, np.vstack(ref_traj))
        self.opti.set_value(self.P_v_tgt, v_target)
        self.opti.set_value(self.P_Ae, Ae)
        self.opti.set_value(self.P_Be, Be)
        self.opti.set_value(self.P_Ce, Ce)
        
        # Pass previous control for damping calculation
        self.opti.set_value(self.P_u_prev, u_prev)
        
        # Warm start
        self.opti.set_initial(self.X, np.repeat(x0.reshape(-1,1), self.N+1, axis=1))

        try:
            sol = self.opti.solve()
            # Return full horizon for visualization
            return sol.value(self.U)[:, 0], sol.value(self.X)
        except Exception as e:
            # Fallback: simple braking
            print(f"Solver Failed: {e}")
            return np.array([-1.0, 0.0]), np.repeat(x0.reshape(-1,1), self.N+1, axis=1)

# ==============================================================================
# 4. ROS 2 NODE WRAPPER
# ==============================================================================
class LMPCNode(Node):
    def __init__(self):
        super().__init__('lmpc_node')
        
        # --- Parameters ---
        self.declare_parameter('track_file', 'raceline.csv')
        self.declare_parameter('drive_topic', '/drive')
        self.declare_parameter('odom_topic', '/odom')
        
        track_file = self.get_parameter('track_file').value
        self.drive_topic = self.get_parameter('drive_topic').value
        self.odom_topic = self.get_parameter('odom_topic').value

        # --- Subsystems ---
        self.track = TrackManager(track_file, W_TRACK)
        self.mpc = CasadiMPC()
        self.learner = LMPCLearner()

        # --- State Variables ---
        # State: [x, y, v, psi]
        self.current_state = None 
        self.prev_state = None
        self.prev_u = np.zeros(2)
        
        # Lap Logic
        self.lap_data = {'X':[], 'U':[], 'X_n':[], 'X_nom':[]}
        self.lap_count = 0
        self.aggressiveness = 0.5
        self.is_racing = False
        
        # --- ROS Interfaces ---
        self.sub_odom = self.create_subscription(
            Odometry, self.odom_topic, self.odom_callback, 10)
        
        self.pub_drive = self.create_publisher(
            AckermannDriveStamped, self.drive_topic, 10)
            
        self.pub_viz_path = self.create_publisher(
            Marker, '/mpc_path', 10)
        self.pub_ref_path = self.create_publisher(
            Marker, '/reference_path', 10)

        # Control Loop Timer
        self.timer = self.create_timer(DT, self.control_loop)
        
        self.get_logger().info("LMPC Node Initialized. Waiting for Odom...")

    def odom_callback(self, msg):
        # Extract State
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        
        # Quaternion to Yaw
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        # Velocity
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        v = np.sqrt(vx**2 + vy**2)
        if vx < 0: v = -v # handle reversing
        
        self.current_state = np.array([px, py, v, yaw])

    def get_nominal_dynamics(self, x, u):
        # The internal model used to calculate Model Mismatch
        x_next = np.zeros(4)
        beta = np.arctan(0.5 * np.tan(u[1]))
        x_next[0] = x[0] + x[2] * np.cos(x[3] + beta) * DT
        x_next[1] = x[1] + x[2] * np.sin(x[3] + beta) * DT
        x_next[2] = x[2] + u[0] * DT
        x_next[3] = x[3] + (x[2] / L_VEHICLE) * np.sin(beta) * DT
        return x_next

    def control_loop(self):
        if self.current_state is None:
            return

        x = self.current_state
        
        # 1. Update Learner (Compare Real(t) vs Prediction(t-1))
        if self.prev_state is not None and self.is_racing:
            # We applied prev_u at prev_state. 
            # We expected nominal_next. We got current_state (real).
            x_nom = self.get_nominal_dynamics(self.prev_state, self.prev_u)
            
            self.lap_data['X'].append(self.prev_state)
            self.lap_data['U'].append(self.prev_u)
            self.lap_data['X_n'].append(x)      # Real outcome
            self.lap_data['X_nom'].append(x_nom)# Physics prediction
            
            # Online model update (every 10 steps)
            if len(self.lap_data['X']) % 10 == 0:
                self.learner.update_error_model(x)

        # 2. Get Reference
        rx, ry, rpsi, v_max, idx = self.track.get_reference(x[0], x[1], N_HORIZON)
        
        # Check Lap Completion (Using Track Index)
        # If we just wrapped around from high index to low index
        if self.lap_count > 0 and idx < 20 and len(self.lap_data['X']) > 200:
             self.finish_lap()
        elif idx > self.track.N_points * 0.95 and not self.is_racing:
             self.is_racing = True # Started the lap
             self.get_logger().info("Lap Started!")

        # 3. Solve MPC
        v_tgt = v_max * self.aggressiveness
        
        # --- MODIFIED: Pass self.prev_u into solve ---
        u_opt, x_pred_traj = self.mpc.solve(
            x, [rx, ry, rpsi], v_tgt, 
            self.learner.Ae, self.learner.Be, self.learner.Ce,
            self.prev_u 
        )

        # 4. Publish Control
        acc = float(u_opt[0])
        steer = float(u_opt[1])
        
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = "base_link"
        drive_msg.drive.steering_angle = steer
        drive_msg.drive.acceleration = acc
        # Simple Speed logic integration for low-level controllers that need speed, not acc
        drive_msg.drive.speed = x[2] + acc * DT 
        self.pub_drive.publish(drive_msg)

        # 5. Visualize
        self.publish_visualization(x_pred_traj, rx, ry)
        
        # 6. Store History
        self.prev_state = x.copy()
        self.prev_u = u_opt.copy()

    def finish_lap(self):
        self.get_logger().info(f"Lap {self.lap_count} Complete! Processing Data...")
        
        X_arr = np.array(self.lap_data['X'])
        if len(X_arr) > 50:
            self.learner.add_data(
                X_arr, 
                np.array(self.lap_data['U']), 
                np.array(self.lap_data['X_n']), 
                np.array(self.lap_data['X_nom'])
            )
            self.lap_count += 1
            self.aggressiveness = min(0.9, self.aggressiveness + 0.1)
            self.get_logger().info(f"Aggressiveness increased to {self.aggressiveness}")
        
        # Reset Data
        self.lap_data = {'X':[], 'U':[], 'X_n':[], 'X_nom':[]}
        self.is_racing = False # Wait until we cross start line logic again

    def publish_visualization(self, pred_traj, rx, ry):
        # 1. MPC Predicted Trajectory (Line Strip)
        msg = Marker()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.type = Marker.LINE_STRIP
        msg.action = Marker.ADD
        msg.scale.x = 0.1 # Line width
        msg.color.a = 1.0; msg.color.g = 1.0 # Green
        
        for i in range(pred_traj.shape[1]):
            p = Point()
            p.x = float(pred_traj[0, i])
            p.y = float(pred_traj[1, i])
            msg.points.append(p)
        self.pub_viz_path.publish(msg)

        # 2. Reference Points (Points)
        ref_msg = Marker()
        ref_msg.header.frame_id = "map"
        ref_msg.header.stamp = self.get_clock().now().to_msg()
        ref_msg.type = Marker.POINTS
        ref_msg.action = Marker.ADD
        ref_msg.scale.x = 0.2; ref_msg.scale.y = 0.2
        ref_msg.color.a = 1.0; ref_msg.color.r = 1.0 # Red
        
        for i in range(len(rx)):
            p = Point()
            p.x = float(rx[i])
            p.y = float(ry[i])
            ref_msg.points.append(p)
        self.pub_ref_path.publish(ref_msg)

def main(args=None):
    rclpy.init(args=args)
    node = LMPCNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop command on exit
        stop_msg = AckermannDriveStamped()
        node.pub_drive.publish(stop_msg)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()