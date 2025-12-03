import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import casadi as ca
from scipy.spatial import KDTree
from scipy.interpolate import splprep, splev
import math
import os
import sys

# --- TUNING ---
DT = 0.05
N_HORIZON = 10
L_VEHICLE = 0.33
FILENAME = 'raceline_2.csv'

# --- CRITICAL CONFIG ---
# Try -1.0 if car spins. Try 1.0 if it steers normally.
STEER_GAIN = -1.0  

# --- ALPP SETTINGS ---
ALPP_MAX_SPEED = 2.0
ALPP_LOOKAHEAD_MIN = 1.0
ALPP_LOOKAHEAD_MAX = 2.5

def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def euler_to_quaternion(yaw):
    return [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]

class TrackManager:
    def __init__(self, filename, logger):
        self.logger = logger
        if not os.path.exists(filename): filename = os.path.join(os.getcwd(), filename)
        
        try:
            raw_data = np.loadtxt(filename, delimiter=',', skiprows=1)
            raw_path = raw_data[:, :2]
            
            # Interpolation (10cm resolution)
            diffs = np.linalg.norm(np.diff(raw_path, axis=0), axis=1)
            valid = np.insert(diffs > 0.01, 0, True)
            clean = raw_path[valid]
            
            tck, u = splprep(clean.T, s=0, k=3, per=1)
            u_new = np.linspace(u.min(), u.max(), int(np.sum(diffs)/0.1))
            x_new, y_new = splev(u_new, tck)
            
            self.path = np.vstack((x_new, y_new)).T
            self.N_points = len(self.path)
            
            # Headings
            dx = np.gradient(self.path[:, 0])
            dy = np.gradient(self.path[:, 1])
            self.headings = np.arctan2(dy, dx)
            
            self.tree = KDTree(self.path)
            self.logger.info(f"TRACK LOADED: {self.N_points} points.")
            
        except Exception as e:
            self.logger.error(f"TRACK FAIL: {e}")
            theta = np.linspace(0, 2*np.pi, 200)
            self.path = np.vstack((10 * np.cos(theta), 6 * np.sin(theta))).T
            self.N_points = 200
            self.headings = theta + np.pi/2
            self.tree = KDTree(self.path)

    def get_reference(self, x, y, last_idx):
        # Window Search
        best_dist = float('inf'); best_idx = last_idx
        
        for i in range(200): # Look 20m ahead
            idx = (last_idx + i) % self.N_points
            d = (self.path[idx,0]-x)**2 + (self.path[idx,1]-y)**2
            if d < best_dist: best_dist = d; best_idx = idx
        
        # Global Rescue if lost
        if best_dist > 25.0: # >5m error
             dist, idx = self.tree.query([x, y])
             best_idx = idx
             best_dist = dist**2

        ref_x, ref_y, ref_psi = [], [], []
        curr = best_idx
        for _ in range(N_HORIZON + 1):
            ref_x.append(self.path[curr, 0])
            ref_y.append(self.path[curr, 1])
            ref_psi.append(self.headings[curr])
            curr = (curr + 2) % self.N_points
            
        return np.array(ref_x), np.array(ref_y), np.array(ref_psi), 2.0, best_idx, math.sqrt(best_dist)

class AdaptivePurePursuitController:
    def __init__(self, track):
        self.track = track; self.wheelbase = L_VEHICLE
        self.max_speed = ALPP_MAX_SPEED
    
    def compute(self, x_curr, last_idx):
        px, py, v, yaw = x_curr
        L = np.clip(1.0 + (v-0.5)*1.0, 1.0, 2.5)
        goal = None
        for i in range(last_idx, last_idx + 500):
            idx = i % self.track.N_points
            pt = self.track.path[idx]
            dist = math.hypot(pt[0]-px, pt[1]-py)
            dx_local = (pt[0]-px)*math.cos(-yaw) - (pt[1]-py)*math.sin(-yaw)
            if dist > L and dx_local > 0: goal = pt; break
        if goal is None: goal = self.track.path[(last_idx+50)%self.track.N_points]
        alpha = normalize_angle(math.atan2(goal[1]-py, goal[0]-px) - yaw)
        steer = math.atan2(2*self.wheelbase*math.sin(alpha), L)
        return 1.0, float(steer * STEER_GAIN), goal

class LMPCLearner:
    def __init__(self, logger):
        self.logger = logger
        self.X_feat = None; self.Y_err = None; self.tree = None
        self.Ae = np.zeros((2, 1)); self.Be = np.zeros((2, 2)); self.Ce = np.zeros((2, 1))

    def add_data(self, X_log, U_log, X_next_log, X_nom_log):
        # Errors [v, yaw]
        global_err = X_next_log - X_nom_log
        v_err = global_err[:, 2:3]
        yaw_err = global_err[:, 3:4]
        
        feats = X_log[:, 2:3] # Velocity only
        targets = np.hstack((v_err, yaw_err))
        
        if self.X_feat is None:
            self.X_feat = feats; self.Y_err = targets; self.U_log = U_log; self.X_search = X_log[:, :2]
        else:
            self.X_feat = np.vstack((self.X_feat, feats))
            self.Y_err = np.vstack((self.Y_err, targets))
            self.U_log = np.vstack((self.U_log, U_log))
            self.X_search = np.vstack((self.X_search, X_log[:, :2]))

        if self.X_feat.shape[0] > 6000:
            idx = np.random.choice(self.X_feat.shape[0], 5000, replace=False)
            self.X_feat = self.X_feat[idx]; self.Y_err = self.Y_err[idx]
            self.U_log = self.U_log[idx]; self.X_search = self.X_search[idx]
        self.tree = KDTree(self.X_search)

    def update_error_model(self, x_curr):
        if self.X_feat is None: return
        dist, idx = self.tree.query(x_curr[:2], k=30)
        h = 3.0; w = np.where(dist/h < 1, 0.75*(1-(dist/h)**2), 0.0)
        
        if np.sum(w) < 1e-3: 
            self.Ae *= 0; self.Be *= 0; self.Ce *= 0; return 
        W = np.diag(w)
        Z = np.hstack((self.X_feat[idx], self.U_log[idx], np.ones((len(idx), 1))))
        E = self.Y_err[idx]
        try:
            Theta = (np.linalg.inv(Z.T @ W @ Z + 1e-3*np.eye(Z.shape[1])) @ Z.T @ W @ E).T
            if np.isnan(Theta).any(): return
            alpha = 0.2
            self.Ae = (1-alpha)*self.Ae + alpha*Theta[:, 0:1]
            self.Be = (1-alpha)*self.Be + alpha*Theta[:, 1:3]
            self.Ce = (1-alpha)*self.Ce + alpha*Theta[:, 3:4]
        except: pass

class CasadiMPC:
    def __init__(self, logger):
        self.logger = logger
        self.opti = ca.Opti(); self.N = N_HORIZON
        self.X = self.opti.variable(4, self.N+1)
        self.U = self.opti.variable(2, self.N)
        self.slack = self.opti.variable(self.N+1)
        
        self.P_x0 = self.opti.parameter(4); self.P_ref = self.opti.parameter(3, self.N+1)
        self.P_v_tgt = self.opti.parameter(1); self.P_u_prev = self.opti.parameter(2)
        self.P_Ae = self.opti.parameter(2,1); self.P_Be = self.opti.parameter(2,2); self.P_Ce = self.opti.parameter(2,1)
        self.P_learn_w = self.opti.parameter(1)

        cost = 0
        for k in range(self.N):
            dist_sq = (self.X[0,k]-self.P_ref[0,k])**2 + (self.X[1,k]-self.P_ref[1,k])**2
            err_vel = (self.X[2,k]-self.P_v_tgt)**2
            if k == 0: du = (self.U[:,k]-self.P_u_prev)**2
            else: du = (self.U[:,k]-self.U[:,k-1])**2
            cost += 10.0*dist_sq + 1.0*err_vel + 500.0*du[1] + 10.0*du[0] + 5.0*self.U[1,k]**2
        cost += 100000.0 * ca.sumsqr(self.slack)
        self.opti.minimize(cost)

        for k in range(self.N):
            x, u = self.X[:,k], self.U[:,k]
            # APPLY STEER GAIN IN SOLVER TOO
            steer_eff = u[1] * STEER_GAIN 
            beta = ca.atan(0.5*ca.tan(steer_eff))
            
            x_nom_0 = x[0]+x[2]*ca.cos(x[3]+beta)*DT
            x_nom_1 = x[1]+x[2]*ca.sin(x[3]+beta)*DT
            x_nom_2 = x[2]+u[0]*DT
            x_nom_3 = x[3]+(x[2]/L_VEHICLE)*ca.sin(beta)*DT
            
            err = ca.mtimes(self.P_Ae, x[2]) + ca.mtimes(self.P_Be, u) + self.P_Ce
            x_next_2 = x_nom_2 + err[0]*self.P_learn_w
            x_next_3 = x_nom_3 + err[1]*self.P_learn_w
            
            self.opti.subject_to(self.X[:,k+1] == ca.vertcat(x_nom_0, x_nom_1, x_next_2, x_next_3))
            
            dist_sq = (x[0]-self.P_ref[0,k])**2 + (x[1]-self.P_ref[1,k])**2
            self.opti.subject_to(dist_sq <= 9.0 + self.slack[k])

        self.opti.subject_to(self.X[:,0]==self.P_x0)
        self.opti.subject_to(self.slack>=0)
        self.opti.subject_to(self.opti.bounded(-5.0, self.U[0,:], 5.0))
        self.opti.subject_to(self.opti.bounded(-0.6, self.U[1,:], 0.6))
        self.opti.subject_to(self.opti.bounded(-2.0, self.X[2,:], 20.0))
        self.opti.solver('ipopt', {'ipopt.print_level':0, 'print_time':0, 'ipopt.max_iter':80, 'ipopt.warm_start_init_point':'yes'})

    def safe_set(self, param, value, shape=None):
        if np.any(np.isnan(value)) or np.any(np.isinf(value)):
            if shape: self.opti.set_value(param, np.zeros(shape))
            else: self.opti.set_value(param, 0.0)
        else:
            self.opti.set_value(param, value)

    def solve(self, x0, ref, v, Ae, Be, Ce, u_prev, w):
        if np.isnan(x0).any(): return None
        self.safe_set(self.P_x0, x0); self.safe_set(self.P_ref, np.vstack(ref))
        self.safe_set(self.P_v_tgt, v); self.safe_set(self.P_u_prev, u_prev)
        self.safe_set(self.P_Ae, Ae, (2,1)); self.safe_set(self.P_Be, Be, (2,2)); self.safe_set(self.P_Ce, Ce, (2,1))
        self.safe_set(self.P_learn_w, w)
        
        x_guess = np.zeros((4, self.N+1)); x_guess[:, 0] = x0; curr = x0.copy()
        for k in range(self.N):
            beta = math.atan(0.5*math.tan(u_prev[1]*STEER_GAIN))
            curr[0]+=curr[2]*math.cos(curr[3]+beta)*DT; curr[1]+=curr[2]*math.sin(curr[3]+beta)*DT
            curr[2]+=u_prev[0]*DT; curr[3]+=(curr[2]/L_VEHICLE)*math.sin(beta)*DT
            x_guess[:,k+1] = curr
        self.opti.set_initial(self.X, x_guess)
        self.opti.set_initial(self.U, np.tile(u_prev.reshape(2,1), (1, self.N)))
        self.opti.set_initial(self.slack, 0.0)
        
        try:
            sol = self.opti.solve()
            return sol.value(self.U)[:,0], sol.value(self.X)
        except: return None

class LMPCNode(Node):
    def __init__(self):
        super().__init__('lmpc_node')
        self.sub = self.create_subscription(Point, '/autodrive/f1tenth_1/ips', self.cb_pose, 10)
        self.pub_th = self.create_publisher(Float32, '/autodrive/f1tenth_1/throttle_command', 10)
        self.pub_st = self.create_publisher(Float32, '/autodrive/f1tenth_1/steering_command', 10)
        self.pub_viz = self.create_publisher(Marker, '/mpc_path', 10)
        self.pub_ref = self.create_publisher(Marker, '/mpc_ref', 10)
        self.pub_arr = self.create_publisher(MarkerArray, '/ref_arrows', 10)
        self.pub_walls = self.create_publisher(Marker, '/track_walls', 10)
        self.pub_viz_goal = self.create_publisher(Marker, '/alpp_goal', 10)
        
        self.track = TrackManager(FILENAME, self.get_logger())
        self.mpc = CasadiMPC(self.get_logger())
        self.learner = LMPCLearner(self.get_logger())
        self.alpp = AdaptivePurePursuitController(self.track)
        
        self.curr_x = None; self.prev_x = None; self.prev_u = np.zeros(2)
        self.lap_data = {'X':[], 'U':[], 'X_n':[], 'X_nom':[]}
        self.lap_count = 0; self.is_racing = False; self.last_idx = 0
        self.create_timer(DT, self.control_loop)
        self.publish_walls()
        self.get_logger().info(f"LMPC Ready. STEER_GAIN={STEER_GAIN}")

    def cb_pose(self, msg):
        if hasattr(msg, 'pose'): px=msg.pose.pose.position.x; py=msg.pose.pose.position.y
        else: px=msg.x; py=msg.y
        if self.prev_x is None: yaw=0.0; v=0.0
        else:
            dx=px-self.prev_x[0]; dy=py-self.prev_x[1]
            yaw=math.atan2(dy, dx) if math.hypot(dx,dy)>0.02 else self.prev_x[3]
            v = 0.6*math.hypot(dx/DT, dy/DT) + 0.4*self.prev_x[2]
        self.curr_x = np.array([px, py, v, yaw])

    def control_loop(self):
        if self.curr_x is None: return
        x = self.curr_x.copy()
        
        rx, ry, rpsi, v_max, idx, dist = self.track.get_reference(x[0], x[1], self.last_idx)
        self.last_idx = idx 
        self.viz_ref(rx, ry); self.viz_ref_arrows(rx, ry, rpsi)

        diff = normalize_angle(x[3] - rpsi[0])
        rpsi[0] = x[3] - diff
        for i in range(1, len(rpsi)):
             delta = normalize_angle(rpsi[i] - rpsi[i-1])
             rpsi[i] = rpsi[i-1] + delta

        if not self.is_racing and idx > 50: self.is_racing = True
        
        is_moving = x[2]>0.1
        if self.prev_x is not None and self.is_racing and is_moving:
            beta = math.atan(0.5*math.tan(self.prev_u[1]*STEER_GAIN))
            x_nom = np.zeros(4)
            x_nom[0] = self.prev_x[0] + self.prev_x[2]*math.cos(self.prev_x[3]+beta)*DT
            x_nom[1] = self.prev_x[1] + self.prev_x[2]*math.sin(self.prev_x[3]+beta)*DT
            x_nom[2] = self.prev_x[2] + self.prev_u[0]*DT
            x_nom[3] = self.prev_x[3] + (self.prev_x[2]/L_VEHICLE)*math.sin(beta)*DT
            self.lap_data['X'].append(self.prev_x); self.lap_data['U'].append(self.prev_u)
            self.lap_data['X_n'].append(x); self.lap_data['X_nom'].append(x_nom)
            if len(self.lap_data['X'])%10==0: self.learner.update_error_model(x)

        if self.is_racing and idx < 20 and len(self.lap_data['X']) > 500 and is_moving: self.finish_lap()

        u_acc, u_steer = 0.0, 0.0
        
        # SAFETY PILOT: If too far off track, override MPC with ALPP
        safety_override = False
        if dist > 1.0 and self.lap_count > 0:
            self.get_logger().warn(f"Off Track ({dist:.2f}m) -> Safety Pilot (ALPP)")
            safety_override = True

        if self.lap_count == 0 or safety_override:
            u_acc, u_steer, goal = self.alpp.compute(x, self.last_idx)
            if goal is not None: self.viz_sphere(goal)
        else:
            v_tgt = min(x[2] + 1.0, v_max * min(1.0, 0.6 + 0.1*self.lap_count))
            w = 1.0 if self.lap_count > 0 else 0.0
            
            res = self.mpc.solve(x, [rx, ry, rpsi], v_tgt, self.learner.Ae, self.learner.Be, self.learner.Ce, self.prev_u, w)
            if res is not None:
                u_acc, u_steer = res[0], res[1]
                self.viz(res[1])
            else:
                self.get_logger().error("SOLVER FAIL -> Fallback ALPP")
                u_acc, u_steer, _ = self.alpp.compute(x, self.last_idx)

        # APPLY STEERING GAIN (INVERT IF NECESSARY)
        final_steer = float(u_steer * STEER_GAIN)
        
        self.pub_th.publish(Float32(data=float(u_acc)))
        self.pub_st.publish(Float32(data=final_steer))
        self.prev_x = x.copy(); self.prev_u = np.array([u_acc, u_steer])

    def finish_lap(self):
        self.lap_count += 1
        if len(self.lap_data['X']) > 50:
            self.learner.add_data(np.array(self.lap_data['X']), np.array(self.lap_data['U']), np.array(self.lap_data['X_n']), np.array(self.lap_data['X_nom']))
        self.lap_data = {'X':[], 'U':[], 'X_n':[], 'X_nom':[]}
        self.last_idx = 0
        self.get_logger().info(f"Lap {self.lap_count} Start")

    def viz(self, pred):
        m = Marker(); m.header.frame_id='world'; m.type=Marker.LINE_STRIP; m.scale.x=0.1; m.color.a=1.0; m.color.r=1.0
        for i in range(pred.shape[1]): m.points.append(Point(x=float(pred[0,i]), y=float(pred[1,i])))
        self.pub_viz.publish(m)

    def viz_ref(self, rx, ry):
        m = Marker(); m.header.frame_id='world'; m.type=Marker.POINTS; m.scale.x=0.2; m.scale.y=0.2; m.color.a=1.0; m.color.b=1.0
        for i in range(len(rx)): m.points.append(Point(x=float(rx[i]), y=float(ry[i])))
        self.pub_ref.publish(m)

    def viz_ref_arrows(self, rx, ry, rpsi):
        ma = MarkerArray()
        del_m = Marker(); del_m.action = Marker.DELETEALL
        ma.markers.append(del_m)
        for i in range(0, len(rx), 2):
            m = Marker(); m.header.frame_id='world'; m.type=Marker.ARROW; m.id=i+100; m.action=Marker.ADD
            m.scale.x=0.5; m.scale.y=0.1; m.scale.z=0.1; m.color.a=1.0; m.color.b=1.0
            m.pose.position.x=rx[i]; m.pose.position.y=ry[i]
            q = euler_to_quaternion(rpsi[i])
            m.pose.orientation.x=q[0]; m.pose.orientation.y=q[1]; m.pose.orientation.z=q[2]; m.pose.orientation.w=q[3]
            ma.markers.append(m)
        self.pub_arr.publish(ma)

    def viz_sphere(self, pt):
        m = Marker(); m.header.frame_id='world'; m.type=Marker.SPHERE; m.action=Marker.ADD
        m.scale.x=0.3; m.scale.y=0.3; m.scale.z=0.3; m.color.a=1.0; m.color.g=1.0
        m.pose.position.x=pt[0]; m.pose.position.y=pt[1]
        self.pub_viz_goal.publish(m)

    def publish_walls(self):
        m = Marker(); m.header.frame_id = 'world'; m.type = Marker.LINE_LIST
        m.scale.x = 0.1; m.color.a = 0.5; m.color.r = 1.0; m.color.g = 1.0
        for i in range(len(self.track.path)-1):
            p1 = self.track.path[i]; p2 = self.track.path[i+1]
            dx = p2[0]-p1[0]; dy = p2[1]-p1[1]; yaw = math.atan2(dy, dx)
            rx = p1[0] + math.sin(yaw)*2.0; ry = p1[1] - math.cos(yaw)*2.0
            lx = p1[0] - math.sin(yaw)*2.0; ly = p1[1] + math.cos(yaw)*2.0
            m.points.append(Point(x=rx, y=ry, z=0.0)); m.points.append(Point(x=rx, y=ry, z=0.5))
            m.points.append(Point(x=lx, y=ly, z=0.0)); m.points.append(Point(x=lx, y=ly, z=0.5))
        self.pub_walls.publish(m)

def main():
    rclpy.init(); node = LMPCNode()
    try: rclpy.spin(node)
    except: pass
    node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()