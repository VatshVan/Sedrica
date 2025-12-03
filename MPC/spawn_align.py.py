import numpy as np
import os

# --- CONFIGURATION ---
INPUT_FILE = 'MPC/raceline_circle.csv'      # Your current track
OUTPUT_FILE = 'MPC/raceline_spawn.csv' # The fixed track

# YOUR EXACT SPAWN POINT
TARGET_X = 0.74
TARGET_Y = 3.16
# ---------------------

def align_track():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    print(f"Loading {INPUT_FILE}...")
    try:
        # Try loading with header skipping
        data = np.loadtxt(INPUT_FILE, delimiter=',', skiprows=1)
    except:
        # Fallback
        data = np.loadtxt(INPUT_FILE, skiprows=1)

    # 1. Get current start
    start_x = data[0, 0]
    start_y = data[0, 1]
    
    # 2. Calculate Shift
    shift_x = TARGET_X - start_x
    shift_y = TARGET_Y - start_y
    
    print(f"Original Start: ({start_x:.2f}, {start_y:.2f})")
    print(f"Target Spawn:   ({TARGET_X:.2f}, {TARGET_Y:.2f})")
    print(f"--> Shifting Track by: X={shift_x:.2f}, Y={shift_y:.2f}")

    # 3. Apply Shift (Preserving Widths in cols 2,3)
    data[:, 0] += shift_x
    data[:, 1] += shift_y

    # 4. Save
    header = "x,y,w_right,w_left"
    np.savetxt(OUTPUT_FILE, data, delimiter=',', header=header, comments='', fmt='%.4f')
    print(f"✅ Success! Saved to '{OUTPUT_FILE}'.")
    print("Update your LMPC code to use this new file.")

if __name__ == "__main__":
    align_track()