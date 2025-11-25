# import pandas as pd
# import numpy as np
# import pysindy as ps
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pickle
# # NEW: Import the scaler
# from sklearn.preprocessing import StandardScaler

# # === 1. Load Data ===
# try:
#     df = pd.read_csv('sindy_training_data.csv')
#     print("Loaded 'sindy_training_data.csv'")
# except FileNotFoundError:
#     raise SystemExit("Error: 'sindy_training_data.csv' not found!")

# # State and control variable names
# state_names = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'u', 'v', 'w', 'p', 'q', 'r']
# control_input_names = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']

# # Convert to numpy arrays
# t = df['time'].to_numpy()
# x = df[state_names].to_numpy()
# u = df[control_input_names].to_numpy()

# print(f"Data shapes: x={x.shape}, u={u.shape}")

# # === 2. Calculate Derivatives ===
# print("\nCalculating derivatives...")
# differentiation_method = ps.FiniteDifference(order=2)
# x_dot = differentiation_method(x, t=t)
# print(f"Calculated derivatives shape: x_dot={x_dot.shape}")


# # === 3. Scale the Data ===
# # NEW: This is the crucial step to ensure all features have a similar magnitude.
# print("\nScaling data...")
# # Scale the states and control inputs
# x_scaler = StandardScaler()
# x_scaled = x_scaler.fit_transform(x)

# u_scaler = StandardScaler()
# u_scaled = u_scaler.fit_transform(u)

# # Scale the derivatives using their own scaler
# x_dot_scaler = StandardScaler()
# x_dot_scaled = x_dot_scaler.fit_transform(x_dot)
# print("Data scaling complete.")


# # === 4. Define SINDy Model ===
# feature_library = ps.PolynomialLibrary(degree=1) + ps.FourierLibrary(n_frequencies=1)

# # NEW: Lower the threshold to avoid eliminating physically important terms
# optimizer = ps.STLSQ(threshold=0.05, alpha=15.0)

# model = ps.SINDy(
#     feature_library=feature_library,
#     optimizer=optimizer,
#     feature_names=state_names + control_input_names,
#     discrete_time=False
# )

# # === 5. Train Model ===
# print("\nTraining SINDy model on SCALED data...")
# # NEW: Use the scaled variables for training
# model.fit(x_scaled, u=u_scaled, x_dot=x_dot_scaled)
# print("Training complete.\n")

# # === 6. Print and Save Results ===
# print("Learned Dynamic Equations (in scaled coordinates):")
# model.print()
# print("\nNote: The coefficients above are for the scaled data.")
# print("To get coefficients in original physical units, they would need to be un-scaled.")

# # Save trained model (optional)
# with open('sindy_model.pkl', 'wb') as f:
#     pickle.dump(model, f)
# print("Model saved to 'sindy_model.pkl'")

# # === 7. Visualize Coefficients ===
# coefficients = model.coefficients()
# library_feature_names = model.get_feature_names()

# plt.figure(figsize=(20, 10))
# sns.heatmap(
#     coefficients,
#     cmap='vlag',
#     annot=False, # Set to True if the heatmap is small enough
#     xticklabels=library_feature_names,
#     yticklabels=[f"{s}_dot" for s in state_names],
#     linewidths=0.5,
#     center=0 # NEW: Center the colormap at zero
# )
# plt.title('SINDy Coefficient Heatmap (Continuous-Time, Scaled Data)', fontsize=16)
# plt.xlabel('Candidate Library Functions')
# plt.ylabel('State Derivatives')
# plt.xticks(rotation=90, ha='right')
# plt.tight_layout()
# plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('data_files/sindy_training_data.csv')

# Figure 1: Forces (Fx, Fy, Fz) vs time
plt.figure(figsize=(10, 6))
plt.plot(df['time'], df['Fx'], label='Fx', linewidth=1.5)
plt.plot(df['time'], df['Fy'], label='Fy', linewidth=1.5)
plt.plot(df['time'], df['Fz'], label='Fz', linewidth=1.5)
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.title('Forces vs Time')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Figure 2: Torques (Tx, Ty, Tz) vs time
plt.figure(figsize=(10, 6))
plt.plot(df['time'], df['Tx'], label='Tx', linewidth=1.5)
plt.plot(df['time'], df['Ty'], label='Ty', linewidth=1.5)
plt.plot(df['time'], df['Tz'], label='Tz', linewidth=1.5)
plt.xlabel('Time (s)')
plt.ylabel('Torque (N·m)')
plt.title('Torques vs Time')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Figure 3: Position and Orientation (x, y, z, roll, pitch, yaw) vs time
fig3, axes3 = plt.subplots(2, 3, figsize=(15, 8))
fig3.suptitle('Position and Orientation vs Time', fontsize=14)

# Flatten axes for easier iteration
axes3 = axes3.flatten()
position_orientation = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
colors3 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
units3 = ['m', 'm', 'm', 'rad', 'rad', 'rad']

for i, var in enumerate(position_orientation):
    axes3[i].plot(df['time'], df[var], linewidth=1.5, color=colors3[i])
    axes3[i].set_xlabel('Time (s)')
    axes3[i].set_ylabel(f'{var} ({units3[i]})')
    axes3[i].set_title(f'{var} vs Time')
    axes3[i].grid(True, alpha=0.3)

plt.tight_layout()

# Figure 4: Velocities (u, v, w, p, q, r) vs time
fig4, axes4 = plt.subplots(2, 3, figsize=(15, 8))
fig4.suptitle('Velocities vs Time', fontsize=14)

# Flatten axes for easier iteration
axes4 = axes4.flatten()
velocities = ['u', 'v', 'w', 'p', 'q', 'r']
colors4 = ['#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#ff9896', '#c49c94']
units4 = ['m/s', 'm/s', 'm/s', 'rad/s', 'rad/s', 'rad/s']

for i, var in enumerate(velocities):
    axes4[i].plot(df['time'], df[var], linewidth=1.5, color=colors4[i])
    axes4[i].set_xlabel('Time (s)')
    axes4[i].set_ylabel(f'{var} ({units4[i]})')
    axes4[i].set_title(f'{var} vs Time')
    axes4[i].grid(True, alpha=0.3)

plt.tight_layout()

# Display all figures
plt.show()