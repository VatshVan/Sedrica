import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# ================= CONFIGURATION =================
FILENAME = 'raceline_circle.csv'
RADIUS = 8.0           # Radius of the center line (Meters)
TRACK_WIDTH = 4.0      # Total width of the track (Meters)
NUM_POINTS = 400       # Resolution (More points = smoother)
START_ANGLE = 0.0      # Start angle in radians (0 is East)
DIRECTION = 1          # 1 for Counter-Clockwise, -1 for Clockwise
# =================================================

def generate_circle():
    # 1. Generate Angles (0 to 2pi)
    # We use endpoint=False so the last point doesn't duplicate the first point perfectly
    theta = np.linspace(START_ANGLE, START_ANGLE + DIRECTION * 2 * np.pi, NUM_POINTS, endpoint=False)

    # 2. Calculate Coordinates (Center Line)
    # x = R * cos(theta)
    # y = R * sin(theta)
    x = RADIUS * np.cos(theta)
    y = RADIUS * np.sin(theta)

    # 3. Calculate Widths
    # Your code expects width from center to wall.
    # So if total is 4.0m, right is 2.0m and left is 2.0m.
    w_half = TRACK_WIDTH / 2.0
    w_right = np.full_like(x, w_half)
    w_left = np.full_like(x, w_half)

    # 4. Create DataFrame
    df = pd.DataFrame({
        'x': x,
        'y': y,
        'w_right': w_right,
        'w_left': w_left
    })

    # 5. Add a Header (Your loader skips row 1)
    # We add a dummy header row, or just save with header=True
    df.to_csv(FILENAME, index=False)
    print(f"✅ Generated '{FILENAME}' with {NUM_POINTS} waypoints.")
    print(f"   Radius: {RADIUS}m | Width: {TRACK_WIDTH}m")

    # --- OPTIONAL: VISUALIZATION ---
    plt.figure(figsize=(8,8))
    plt.title(f"Generated Circular Track (R={RADIUS}m)")
    plt.axis('equal')
    
    # Plot Center
    plt.plot(x, y, 'g--', label='Center Line (Reference)')
    
    # Plot Boundaries
    # Inner Wall
    xi = (RADIUS - w_half) * np.cos(theta)
    yi = (RADIUS - w_half) * np.sin(theta)
    # Outer Wall
    xo = (RADIUS + w_half) * np.cos(theta)
    yo = (RADIUS + w_half) * np.sin(theta)
    
    plt.plot(xi, yi, 'k-', linewidth=2, label='Walls')
    plt.plot(xo, yo, 'k-', linewidth=2)
    
    # Plot Start Point
    plt.plot(x[0], y[0], 'ro', label='Start Point')
    
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    generate_circle()