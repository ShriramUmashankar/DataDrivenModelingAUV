import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt


dt = 0.1
# === 1. Load SINDy model and scalers ===
try:
    with open("data_files/model_reduced.pkl", "rb") as f:
        sindy_model = pickle.load(f)
    with open("data_files/scalers_reduced.pkl", "rb") as f:
        scalers = pickle.load(f)
except FileNotFoundError as e:
    print(f"Error: Could not load model or scalers file ({e.filename}).")
    exit()

x_scaler = scalers["x_scaler"]
u_scaler = scalers["u_scaler"]

print("for X the mean and scale is:", x_scaler.mean_, x_scaler.scale_)
print("for U the mean and scale is:", u_scaler.mean_, u_scaler.scale_)
print("Model and original scalers loaded successfully.")
sindy_model.print()

# === 2. Load DMDc matrices ===
A = np.load("data_files/A_tilde.npy")
B = np.load("data_files/B_tilde.npy")
print(f"\nLoaded DMDc matrices: A {A.shape}, B {B.shape}")

# === 3. Load dataset (ground truth and controls) ===
df = pd.read_csv("data_files/sindy_training_data.csv")
df = df.iloc[4000:, :]  # same slice as your SINDy test

gt_vel = df[["u", "v", "w", "p", "q", "r"]].to_numpy()
controls = df[["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]].to_numpy()
t = df["time"].to_numpy()

# === 4. Prepare scaled inputs for SINDy ===
initial_condition = gt_vel[0, :]
initial_condition_scaled = x_scaler.transform(initial_condition.reshape(1, -1)).flatten()
controls_scaled = u_scaler.transform(controls)

# === 5. Run SINDy simulation ===
print("\nRunning SINDy simulation...")
sim_vel_scaled = sindy_model.simulate(
    x0=initial_condition_scaled,
    t=t,
    u=controls_scaled
)
sim_vel_sindy = x_scaler.inverse_transform(sim_vel_scaled)

# === 6. Run DMDc simulation ===
print("Running DMDc simulation...")
x0 = gt_vel[0, :]
x = x0.copy()
n_steps = len(t)
n_state = A.shape[0]
pred_vel_dmdc = np.zeros((n_steps, n_state))
pred_vel_dmdc[0, :] = x0

for k in range(n_steps - 1):
    u_k = controls[k, :]
    x_next = (np.eye(A.shape[0]) + A *dt) @ x + (B @ u_k)*dt
    pred_vel_dmdc[k + 1, :] = x_next
    x = x_next

# === 7. Align lengths ===
n = min(len(gt_vel), len(sim_vel_sindy), len(pred_vel_dmdc))
gt_vel = gt_vel[:n]
sim_vel_sindy = sim_vel_sindy[:n]
pred_vel_dmdc = pred_vel_dmdc[:n]
t = t[:n]

# === 8. Compute metrics ===
def compute_metrics(pred, gt):
    err = pred - gt
    rmse = np.sqrt(np.mean(err**2, axis=0))
    mae = np.mean(np.abs(err), axis=0)
    return rmse, mae

rmse_sindy, mae_sindy = compute_metrics(sim_vel_sindy, gt_vel)
rmse_dmdc, mae_dmdc = compute_metrics(pred_vel_dmdc, gt_vel)
labels = ["u", "v", "w", "p", "q", "r"]

print("\n--- Model Performance ---")
print(f"{'State':<5} | {'SINDy RMSE':<12} | {'DMDc RMSE':<12}")
print("-" * 40)
for i, label in enumerate(labels):
    print(f"{label:<5} | {rmse_sindy[i]:<12.6f} | {rmse_dmdc[i]:<12.6f}")

# === 9. Plot ===
fig, axs = plt.subplots(3, 2, figsize=(14, 11), sharex=True)
for i, ax in enumerate(axs.flat):
    ax.plot(t, gt_vel[:, i], "k-", label="Ground Truth", linewidth=2)
    ax.plot(t, sim_vel_sindy[:, i], "r--", label="SINDy", linewidth=2)
    ax.plot(t, pred_vel_dmdc[:, i], "b-.", label="DMDc", linewidth=2)
    ax.set_ylabel(f"{labels[i]}", fontsize=12)
    ax.set_xlabel("Time")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

plt.suptitle("SINDy vs DMDc vs Ground Truth", fontsize=16, weight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
