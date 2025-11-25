import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# === 1. Load the TRAINED models and ORIGINAL scalers ===
models_files = {
    "Reduced (Order 1)": ("data_files/model_reduced.pkl", "data_files/scalers_reduced.pkl"),
    "Full (Order 2)": ("data_files/model_full.pkl", "data_files/scalers_full.pkl")
}

models = {}
scalers_dict = {}

for key, (model_file, scaler_file) in models_files.items():
    try:
        with open(model_file, "rb") as f:
            models[key] = pickle.load(f)
        with open(scaler_file, "rb") as f:
            scalers_dict[key] = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Error: Could not load {key} ({e.filename}).")
        exit()

print("Models and scalers loaded successfully.\n")

# === 2. Load new dataset ===
df_new = pd.read_csv("data_files/sindy_training_data.csv")
df_new = df_new.iloc[4000:,:]
gt_vel_new = df_new[["u", "v", "w", "p", "q", "r"]].to_numpy()
controls_new = df_new[["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]].to_numpy()
t_new = df_new["time"].to_numpy()

# Prepare dictionaries to hold simulation results and errors
sim_results = {}
errors_dict = {}
rmse_dict = {}
mae_dict = {}

# === 3. Run simulation for each model ===
for key in models.keys():
    x_scaler = scalers_dict[key]["x_scaler"]
    u_scaler = scalers_dict[key]["u_scaler"]

    # Initial condition and scaling
    initial_condition_scaled = x_scaler.transform(gt_vel_new[0, :].reshape(1, -1)).flatten()
    controls_scaled = u_scaler.transform(controls_new)

    # Run simulation
    sim_vel_scaled = models[key].simulate(
        x0=initial_condition_scaled,
        t=t_new,
        u=controls_scaled
    )

    # Inverse transform to physical units
    sim_vel = x_scaler.inverse_transform(sim_vel_scaled)
    sim_len = len(sim_vel)
    gt_trimmed = gt_vel_new[:sim_len]
    t_trimmed = t_new[:sim_len]

    sim_results[key] = sim_vel

    # Compute errors
    errors = sim_vel - gt_trimmed
    errors_dict[key] = errors
    rmse_dict[key] = np.sqrt(np.mean(errors**2, axis=0))
    mae_dict[key] = np.mean(np.abs(errors), axis=0)

# === 4. Print RMSE and MAE side by side ===
labels = ["u (m/s)", "v (m/s)", "w (m/s)", "p (rad/s)", "q (rad/s)", "r (rad/s)"]
print("\n--- Model Performance Comparison ---")
print(f"{'State':<5} | {'RMSE Reduced':<12} | {'RMSE Full':<12} | {'MAE Reduced':<12} | {'MAE Full':<12}")
print("-"*70)
for i, label in enumerate(labels):
    print(f"{label:<5} | {rmse_dict['Reduced (Order 1)'][i]:<12.6f} | "
          f"{rmse_dict['Full (Order 2)'][i]:<12.6f} | {mae_dict['Reduced (Order 1)'][i]:<12.6f} | "
          f"{mae_dict['Full (Order 2)'][i]:<12.6f}")

# === 5. Plot results ===
fig, axs = plt.subplots(3, 2, figsize=(14, 11), sharex=True)
for i, ax in enumerate(axs.flat):
    ax.plot(t_trimmed, gt_trimmed[:, i], "k-", label="Ground Truth", linewidth=2)
    ax.plot(t_trimmed, sim_results["Reduced (Order 1)"][:, i], "r--", label="SINDy Reduced (Order 1)", linewidth=2)
    ax.plot(t_trimmed, sim_results["Full (Order 2)"][:, i], "b--", label="SINDy Full (Order 2)", linewidth=2)
    ax.set_ylabel(f"{labels[i]}", fontsize=12)
    ax.set_xlabel("Time (s)")
    ax.grid(True, linestyle='--', alpha=0.6)
    if i == 0:
        ax.legend()

plt.suptitle("Degree 1 vs 2 polynomial (Sum of Sine input)", fontsize=16, weight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.show()
