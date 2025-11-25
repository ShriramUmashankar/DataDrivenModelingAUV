import pandas as pd
import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler
from finite_differences import SimpleFiniteDifference

# === 1. Load Data ===

df = pd.read_csv('data_files/sindy_training_data.csv')
df = df.iloc[:4000, :]
print("Loaded 'sindy_training_data.csv'")


# === 2. Select Subset of States ===
# Only use body-frame dynamics
state_names = ['u', 'v', 'w', 'p', 'q', 'r']
control_input_names = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']

# Convert to numpy arrays
t = df['time'].to_numpy().flatten()
dt = 0.1

x = df[state_names].to_numpy()
u = df[control_input_names].to_numpy()


print(f"Using reduced states: {state_names}")
print(f"Data shapes: x={x.shape}, u={u.shape}")

# === 3. Compute Derivatives ===
fd = SimpleFiniteDifference(axis=0)
    
x_dot = fd._differentiate(x, dt)

# differentiation_method = ps.FiniteDifference(order=2)
# x_dot = differentiation_method(x)/dt
print(f"Calculated derivatives shape: x_dot={x_dot.shape}")

# === 4. Scale Data ===

x_scaler = StandardScaler()
x_scaled = x_scaler.fit_transform(x)

u_scaler = StandardScaler()
u_scaled = u_scaler.fit_transform(u)

x_dot_scaler = StandardScaler()
x_dot_scaled = x_dot_scaler.fit_transform(x_dot)
print("Data scaling complete.")



# === 5. Define SINDy Model ===
feature_library = ps.PolynomialLibrary(degree = 2)# + ps.FourierLibrary(n_frequencies=1)
optimizer = ps.STLSQ(threshold=0.15, alpha=1.5)

model = ps.SINDy(
    feature_library=feature_library,
    optimizer=optimizer,
    feature_names=state_names + control_input_names,
    discrete_time=False
)

# === 6. Train Model ===
print("\nTraining SINDy model on reduced (scaled) data...")
model.fit(x_scaled, u=u_scaled, x_dot=x_dot_scaled)
print("Training complete.\n")

# === 7. Print and Save Results ===
print("Learned Dynamic Equations (in scaled coordinates):")
model.print()
print("\nNote: The coefficients above are for the scaled data.")
print("To get coefficients in physical units, inverse-transform with the scalers.")

with open('data_files/scalers_full.pkl', 'wb') as f:
    pickle.dump({'x_scaler': x_scaler, 'u_scaler': u_scaler, 'x_dot_scaler': x_dot_scaler}, f)

with open('data_files/model_full.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model saved to 'sindy_model_full.pkl'")

# === 8. Visualize Coefficients ===
coefficients = model.coefficients()
library_feature_names = model.get_feature_names()

plt.figure(figsize=(16, 8))
sns.heatmap(
    coefficients,
    cmap='vlag',
    annot=False,
    xticklabels=library_feature_names,
    yticklabels=[f"{s}_dot" for s in state_names],
    linewidths=0.5,
    center=0
)
plt.title('SINDy Coefficient Heatmap (Reduced States, Scaled Data)', fontsize=16)
plt.xlabel('Candidate Library Functions')
plt.ylabel('State Derivatives')
plt.xticks(rotation=90, ha='right')
plt.tight_layout()
plt.show()
