import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------
# CONFIGURATION
# ---------------------------------------
dt = 0.1                 # time step [s]
total_time = 500         # total simulation duration [s]
t = np.arange(0, total_time, dt)

# Define per-axis amplitudes
amp_fx = 80.0
amp_fy = 80.0
amp_fz = 80.0
amp_tx = 0.0     # roll torque not used
amp_ty = 80.0     # pitch torque small
amp_tz = 80.0     # yaw torque not used

# Frequency range (Hz)
freqs = np.array([0.02, 0.05, 0.1, 0.2])  # covers slow → fast dynamics
phases = np.random.uniform(0, 2*np.pi, len(freqs))

# ---------------------------------------
# FUNCTION TO GENERATE SUM-OF-SINES SIGNAL
# ---------------------------------------
def sum_of_sines(t, amplitude, freqs, phases):
    """Generate sum of sine waves with given frequencies and random phases."""
    signal = np.zeros_like(t)
    for f, p in zip(freqs, phases):
        signal += np.sin(2 * np.pi * f * t + p)
    signal /= len(freqs)  # normalize amplitude
    return amplitude * signal

# ---------------------------------------
# GENERATE FORCES
# ---------------------------------------
Fx = sum_of_sines(t, amp_fx, freqs, phases + 0.1)
Fy = sum_of_sines(t, amp_fy, freqs, phases + 1.1)
Fz = sum_of_sines(t, amp_fz, freqs, phases + 2.2)
Tx = sum_of_sines(t, amp_tx, freqs, phases + 0.3)
Ty = sum_of_sines(t, amp_ty, freqs, phases + 0.6)
Tz = sum_of_sines(t, amp_tz, freqs, phases + 0.9)

# Stack into DataFrame
forces_df = pd.DataFrame({
    "time": t,
    "Fx": Fx,
    "Fy": Fy,
    "Fz": Fz,
    "Tx": Tx,
    "Ty": Ty,
    "Tz": Tz
})

# ---------------------------------------
# SAVE TO CSV
# ---------------------------------------
#forces_df.to_csv("auv_sum_of_sines_forces.csv", index=False)
print("✅ Saved 'auv_sum_of_sines_forces.csv' with shape:", forces_df.shape)

# ---------------------------------------
# PLOT FORCES
# ---------------------------------------
plt.figure(figsize=(10, 7))
plt.plot(t, Fx, label="Fx")
plt.plot(t, Fy, label="Fy")
plt.plot(t, Fz, label="Fz")
plt.xlabel("Time [s]")
plt.ylabel("Force / Torque")
plt.title("Sum-of-Sines Excitation Forces")
plt.legend()
plt.grid(True)
plt.tight_layout()
    

plt.figure(figsize=(10, 7))
plt.plot(t, Tx, label="Tx")
plt.plot(t, Ty, label="Ty")
plt.plot(t, Tz, label="Tz")
plt.xlabel("Time [s]")
plt.ylabel("Force / Torque")
plt.title("Sum-of-Sines Excitation Forces")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

