import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# === 1. Load data ===
try:
    df = pd.read_csv("sindy_training_data.csv")
except FileNotFoundError:
    print("Error: 'sindy_training_data.csv' not found. Please ensure the data file is in the correct directory.")
    exit()

controls = df[["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]].to_numpy()
gt_vel = df[["u", "v", "w", "p", "q", "r"]].to_numpy()
t = df["time"].to_numpy()
dt = t[1] - t[0]
N = len(df)

# === 2. Load model and SCALERS ===
# NOTE: We load the full, original continuous-time model.
try:
    with open("sindy_model_reduced.pkl", "rb") as f:
        sindy_model = pickle.load(f)
    with open("scalers.pkl", "rb") as f:
        scalers = pickle.load(f)
except FileNotFoundError as e:
    print(f"Error: Could not load model or scalers file ({e.filename}). This script requires both 'sindy_model.pkl' and 'scalers.pkl'.")
    exit()

x_scaler = scalers["x_scaler"]
u_scaler = scalers["u_scaler"]


# === 3. Simulate using the model's built-in method ===
print("Starting simulation with model.simulate()...")

# --- Scale the inputs for the model ---
# Define the initial condition from the ground truth data and scale it
initial_condition = gt_vel[0, :]
initial_condition_scaled = x_scaler.transform(initial_condition.reshape(1, -1)).flatten()

# Scale the entire control input sequence
controls_scaled = u_scaler.transform(controls)

# Use the model's simulate function with SCALED data.
sim_vel_scaled = sindy_model.simulate(
    x0=initial_condition_scaled,
    t=t,
    u=controls_scaled,
    integrator_kws={'method': 'Radau'}
)

# --- Inverse transform the simulation output ---
# The output of simulate is in the scaled space, so we transform it back
sim_vel = x_scaler.inverse_transform(sim_vel_scaled)

# --- FIX: Align array lengths for comparison ---
# The ODE solver might return an array that is one sample shorter than the input time array.
# We align the ground truth and time arrays to match the simulation output length.
sim_len = len(sim_vel)
gt_vel = gt_vel[:sim_len]
t = t[:sim_len]


print("Simulation complete.")
print(f"Aligning data for comparison. Using {sim_len} time steps.")


# === 4. Compare with ground truth ===
errors = sim_vel - gt_vel
rmse = np.sqrt(np.mean(errors**2, axis=0))
mae = np.mean(np.abs(errors), axis=0)
labels = ["u", "v", "w", "p", "q", "r"]

print("\n--- Model Performance ---")
print(f"{'State':<5} | {'RMSE':<12} | {'MAE':<12}")
print("-" * 35)
for i, label in enumerate(labels):
    print(f"{label:<5} | {rmse[i]:<12.6f} | {mae[i]:<12.6f}")


# === 5. Plot results ===
fig, axs = plt.subplots(3, 2, figsize=(14, 11), sharex=True)
for i, ax in enumerate(axs.flat):
    ax.plot(t, gt_vel[:, i], "k-", label="Ground Truth", linewidth=2)
    ax.plot(t, sim_vel[:, i], "r--", label="SINDy Simulation", linewidth=2)
    ax.set_ylabel(f"{labels[i]} [m/s or rad/s]", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(labelsize=10)

axs[-1, 0].set_xlabel("Time [s]", fontsize=12)
axs[-1, 1].set_xlabel("Time [s]", fontsize=12)
plt.suptitle("SINDy Continuous-Time Simulation vs. Ground Truth", fontsize=16, weight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

