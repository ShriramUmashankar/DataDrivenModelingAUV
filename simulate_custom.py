import numpy as np
import pandas as pd
from VehicleDynamics import Dynamics
import matplotlib.pyplot as plt
import utils

# ---------------------------------------
# CONFIGURATION
# ---------------------------------------
dt = 0.1  # ensure this matches utils.DT
force_file = "sequential_pulse_forces.csv"
output_file = "sindy_test.csv"

# ---------------------------------------
# INITIALIZE SYSTEM
# ---------------------------------------
amogh = Dynamics('Amogh')


# Load external forces (Fx, Fy, Fz, Tx, Ty, Tz)
force_data = pd.read_csv(force_file)
t = force_data["time"].values
forces = force_data[["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]].values

SIM_STEP = len(t)
print("SIM STEP is", SIM_STEP)

# ---------------------------------------
# STORAGE ARRAYS
# ---------------------------------------
states = []  # [x, y, z, roll, pitch, yaw, u, v, w, p, q, r]


# ---------------------------------------
# SIMULATION LOOP
# ---------------------------------------
for i in range(SIM_STEP):

    if i % 100 ==0:
        print("Percentage done is:", (i*100)/SIM_STEP)
    # Force as 6x1 column vector
    force = forces[i, :].reshape((6, 1))
    
    # Run one simulation step
    solution_matrix, solution_time = amogh.run_simulation(amogh.current_state.flatten(), force)
    
    # Store states
    states.append(amogh.current_state.flatten())


# ---------------------------------------
# SAVE RESULTS
# ---------------------------------------
states = np.round(np.array(states),3)


columns_state = [
    "x", "y", "z", "roll", "pitch", "yaw",
    "u", "v", "w", "p", "q", "r"
]
columns_force = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]


df = pd.DataFrame(
    np.hstack((t.reshape(-1,1), forces, states)),
    columns=["time"] + columns_force + columns_state
)

# df.to_csv(output_file, index=False)
# print(f" Saved SINDy training data to '{output_file}' with shape {df.shape}")

# ---------------------------------------
# OPTIONAL: Quick Plot
# ---------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(t, states[:, 0], label='x')
plt.plot(t, states[:, 1], label='y')
plt.plot(t, states[:, 2], label='yaw')
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("States")
plt.title("AUV Open-Loop Response to Step-Sine Forces")
plt.grid()
plt.tight_layout()
plt.show()
