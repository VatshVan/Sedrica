import numpy as np
import os

# ==========================================
# INPUTS: CHANGE THESE TWO NUMBERS
# ==========================================
# Enter the numbers you saw in the Simulator IPS overlay:
CAR_SPAWN_X = 0.0   # <--- REPLACE WITH YOUR IPS X
CAR_SPAWN_Y = 0.0   # <--- REPLACE WITH YOUR IPS Y
# ==========================================

INPUT_FILE = 'raceline.csv'
OUTPUT_FILE = 'raceline_shifted.csv'

def align_track():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    print(f"Loading {INPUT_FILE}...")
    try:
        # Load all columns (x, y, w_right, w_left)
        data = np.loadtxt(INPUT_FILE, delimiter=',', skiprows=1)
    except:
        try: data = np.loadtxt(INPUT_FILE, skiprows=1)
        except: print("Failed to load CSV."); return

    # 1. Get the track's CURRENT starting point (Row 0)
    track_start_x = data[0, 0]
    track_start_y = data[0, 1]
    
    print(f"Track currently starts at: X={track_start_x:.2f}, Y={track_start_y:.2f}")
    print(f"You want it to start at:   X={CAR_SPAWN_X:.2f},   Y={CAR_SPAWN_Y:.2f}")

    # 2. Calculate the difference (The Shift)
    shift_x = CAR_SPAWN_X - track_start_x
    shift_y = CAR_SPAWN_Y - track_start_y

    print(f"--> Applying Shift: X+{shift_x:.2f}, Y+{shift_y:.2f}")

    # 3. Apply Shift to Global Coordinates (X and Y only)
    data[:, 0] += shift_x  # Shift all X values
    data[:, 1] += shift_y  # Shift all Y values

    # NOTE: We do NOT touch columns 2 and 3 (Widths). 
    # Track width is relative to the center line, so it moves WITH the line automatically.

    # 4. Save the new aligned file
    header = "x,y,w_right,w_left"
    np.savetxt(OUTPUT_FILE, data, delimiter=',', header=header, comments='', fmt='%.4f')
    print(f"\nSUCCESS! Saved to '{OUTPUT_FILE}'.")
    print(f"Update your MPC code to read '{OUTPUT_FILE}'")

if __name__ == "__main__":
    align_track()