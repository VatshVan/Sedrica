import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

N = 1000
t = np.linspace(0, 2*np.pi, N, endpoint=False)

R = 50 + 3.5*np.sin(2*t) + 2.0*np.sin(4*t + 0.6) + 1.2*np.cos(6*t - 0.3)
bulge = 3.2 * np.exp(-((t - 1.7*np.pi) % (2*np.pi) - np.pi)**2 / 0.6)
R = R + bulge

x = R * np.cos(t)
y = R * np.sin(t) * 0.95

dt = t[1] - t[0]
dx = np.gradient(x, dt)
dy = np.gradient(y, dt)
ddx = np.gradient(dx, dt)
ddy = np.gradient(dy, dt)

heading = np.arctan2(dy, dx)
den = (dx**2 + dy**2)**1.5
den[den == 0] = 1e-9
curvature = (dx * ddy - dy * ddx) / den

v_advisory = 12.0 / (np.abs(curvature) + 0.05)
v_advisory = np.clip(v_advisory, 4.0, 40.0)

half_width = 3.5 + 1.2*np.sin(3*t + 0.4)
speed_mag = np.sqrt(dx**2 + dy**2)
tx = dx / (speed_mag + 1e-9)
ty = dy / (speed_mag + 1e-9)
nx = -ty
ny = tx

left_x = x + nx * half_width
left_y = y + ny * half_width
right_x = x - nx * half_width
right_y = y - ny * half_width

df = pd.DataFrame({
    'x': x,
    'y': y,
    'idx': np.arange(N),
    'heading_rad': heading,
    'curvature_1_per_m': curvature,
    'speed_advisory_m_per_s': v_advisory,
    'left_x': left_x,
    'left_y': left_y,
    'right_x': right_x,
    'right_y': right_y
})

df_end = df.iloc[[0]].copy()
df_end['idx'] = N
df2 = pd.concat([df, df_end], ignore_index=True)

out_path = 'MPC/raceline.csv'
df2.to_csv(out_path, index=False)
print(f"Track generated: {out_path}")

plt.figure(figsize=(8, 8))
plt.plot(df2['x'], df2['y'], 'b-', label='Center')
plt.plot(df2['left_x'], df2['left_y'], 'k--', label='Left')
plt.plot(df2['right_x'], df2['right_y'], 'k--', label='Right')
plt.axis('equal')
plt.title("Generated Complex Track")
plt.legend()
plt.grid(True)
plt.savefig("MPC/raceline.png")