import utils
import numpy as np
from VehicleDynamics import Dynamics
from Controller import LQRController
import matplotlib.pyplot as plt


# --- Setup ---
amogh_full = Dynamics('Amogh')
amogh_sindy = Dynamics('Amogh')  # separate copy for independent simulation
LQR = LQRController(Q_scale=6000.0, R_scale=1.0)

print(LQR.m_lqr)
print(utils.zd)

A = np.load('data_files/A.npy')
B = np.load('data_files/B.npy')

# --- Data containers ---
results = {
    "full": {"force_x": [], "force_y": [], "force_z": [], "moment_x": [], "moment_y": [], "moment_z": [],
              "x": [], "y": [], "z": [], "roll": [], "pitch": [], "yaw": []},
    "sindy": {"force_x": [], "force_y": [], "force_z": [], "moment_x": [], "moment_y": [], "moment_z": [],
               "x": [], "y": [], "z": [], "roll": [], "pitch": [], "yaw": []},
}

# --- Simulation loop ---
for i in range(utils.SIM_STEP):
    # === FULL MODEL ===
    eta_full = amogh_full.current_state[:6, :]
    v_full = amogh_full.current_state[6:, :]
    force_full = LQR.compute_lqr_force(
        amogh_full.current_state,
        amogh_full.M,
        amogh_full.D @ np.diagflat(np.abs(v_full)),
        amogh_full.get_C(v_full)
    )
    force_full = np.clip(force_full, -80, 80)
    amogh_full.run_simulation(amogh_full.current_state.flatten(), force_full)

    # Store results (assuming first 3 are forces, last 3 are moments)
    results["full"]["force_x"].append(force_full.flatten()[0])
    results["full"]["force_y"].append(force_full.flatten()[1])
    results["full"]["force_z"].append(force_full.flatten()[2])
    results["full"]["moment_x"].append(force_full.flatten()[3])
    results["full"]["moment_y"].append(force_full.flatten()[4])
    results["full"]["moment_z"].append(force_full.flatten()[5])
    results["full"]["x"].append(amogh_full.current_state.flatten()[0])
    results["full"]["y"].append(amogh_full.current_state.flatten()[1])
    results["full"]["z"].append(amogh_full.current_state.flatten()[2])
    results["full"]["roll"].append(amogh_full.current_state.flatten()[3])
    results["full"]["pitch"].append(amogh_full.current_state.flatten()[4])
    results["full"]["yaw"].append(amogh_full.current_state.flatten()[5])

    # === SINDY MODEL ===
    force_sindy = LQR.compute_lqr_force_sindy(amogh_sindy.current_state)
    force_sindy = np.clip(force_sindy, -80, 80)
    amogh_sindy.run_simulation(amogh_sindy.current_state.flatten(), force_sindy)

    results["sindy"]["force_x"].append(force_sindy.flatten()[0])
    results["sindy"]["force_y"].append(force_sindy.flatten()[1])
    results["sindy"]["force_z"].append(force_sindy.flatten()[2])
    results["sindy"]["moment_x"].append(force_sindy.flatten()[3])
    results["sindy"]["moment_y"].append(force_sindy.flatten()[4])
    results["sindy"]["moment_z"].append(force_sindy.flatten()[5])
    results["sindy"]["x"].append(amogh_sindy.current_state.flatten()[0])
    results["sindy"]["y"].append(amogh_sindy.current_state.flatten()[1])
    results["sindy"]["z"].append(amogh_sindy.current_state.flatten()[2])
    results["sindy"]["roll"].append(amogh_sindy.current_state.flatten()[3])
    results["sindy"]["pitch"].append(amogh_sindy.current_state.flatten()[4])
    results["sindy"]["yaw"].append(amogh_sindy.current_state.flatten()[5])

    if i % 100 == 0:
        print(f"Simulation progress: {(i / utils.SIM_STEP) * 100:.1f} %")


## --- DMDC Model Simulation ---
amogh_dmdc = Dynamics('Amogh')  # independent copy for DMDC test
results["dmdc"] = {"x": [], "y": [], "z": []}

for i in range(utils.SIM_STEP):
    force_dmdc = LQR.compute_lqr_force_dmdc(amogh_dmdc.current_state, A, B)
    force_dmdc = np.clip(force_dmdc, -80, 80)
    amogh_dmdc.run_simulation(amogh_dmdc.current_state.flatten(), force_dmdc)

    results["dmdc"]["x"].append(amogh_dmdc.current_state.flatten()[0])
    results["dmdc"]["y"].append(amogh_dmdc.current_state.flatten()[1])
    results["dmdc"]["z"].append(amogh_dmdc.current_state.flatten()[2])

    if i % 100 == 0:
        print(f"DMDC Simulation progress: {(i / utils.SIM_STEP) * 100:.1f} %")


def plot_comparison(results):
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'lines.linewidth': 2,
        'grid.alpha': 0.3,
        'figure.figsize': (12, 10),
    })

    time = np.arange(len(results["full"]["x"])) * utils.dt  # assuming dt in utils
    line_start = np.array(utils.line_start)
    line_end = np.array(utils.line_end)

    # ---------- FIGURE 1: Position & Orientation ----------
    fig1, axs1 = plt.subplots(2, 2)
    axs1 = axs1.ravel()

    # (a) XY Trajectory
    axs1[0].plot(results["full"]["x"], results["full"]["y"], label='Full', color='C0')
    axs1[0].plot(results["sindy"]["x"], results["sindy"]["y"], '--', label='SINDy', color='C1')
    axs1[0].plot([line_start[0], line_end[0]], [line_start[1], line_end[1]], 'r--', label='Ref line')
    axs1[0].set_title('(a) XY Trajectory')
    axs1[0].set_xlabel('X [m]')
    axs1[0].set_ylabel('Y [m]')
    axs1[0].axis('equal')
    axs1[0].legend()

    # (b) Z vs Time
    axs1[1].plot(time, results["full"]["z"], label='Full', color='C0')
    axs1[1].plot(time, results["sindy"]["z"], '--', label='SINDy', color='C1')
    axs1[1].axhline(utils.zd, color='r', linestyle='--', label='Ref z')
    axs1[1].set_title('(b) Z vs Time')
    axs1[1].set_xlabel('Time [s]')
    axs1[1].set_ylabel('Z [m]')
    axs1[1].legend()

    # (c) Yaw vs Time
    yaw_full = np.array(results["full"]["yaw"]) * 180 / np.pi
    yaw_sindy = np.array(results["sindy"]["yaw"]) * 180 / np.pi
    axs1[2].plot(time, yaw_full, label='Full', color='C0')
    axs1[2].plot(time, yaw_sindy, '--', label='SINDy', color='C1')
    axs1[2].axhline(np.arctan(LQR.m_lqr) * (180 / np.pi), color='r', linestyle='--', label='Ref yaw')
    axs1[2].set_title('(c) Yaw vs Time')
    axs1[2].set_xlabel('Time [s]')
    axs1[2].set_ylabel('Yaw [deg]')
    axs1[2].legend()

    # (d) Pitch & Roll vs Time
    pitch_full = np.array(results["full"]["pitch"]) * 180 / np.pi
    roll_full = np.array(results["full"]["roll"]) * 180 / np.pi
    pitch_sindy = np.array(results["sindy"]["pitch"]) * 180 / np.pi
    roll_sindy = np.array(results["sindy"]["roll"]) * 180 / np.pi

    axs1[3].plot(time, roll_full, label='Roll (Full)', color='C2')
    axs1[3].plot(time, roll_sindy, '--', label='Roll (SINDy)', color='C2', alpha=0.7)
    axs1[3].plot(time, pitch_full, label='Pitch (Full)', color='C3')
    axs1[3].plot(time, pitch_sindy, '--', label='Pitch (SINDy)', color='C3', alpha=0.7)
    axs1[3].set_title('(d) Pitch & Roll vs Time')
    axs1[3].set_xlabel('Time [s]')
    axs1[3].set_ylabel('Angle [deg]')
    axs1[3].legend()

    plt.tight_layout()
    plt.show()

    # ---------- FIGURE 2: Forces & Moments ----------
    fig2, axs2 = plt.subplots(1, 2, figsize=(12, 5))

    # (a) Forces
    axs2[0].plot(time, results["full"]["force_x"], label='Fx (Full)', color='C0')
    axs2[0].plot(time, results["full"]["force_y"], label='Fy (Full)', color='C1')
    axs2[0].plot(time, results["full"]["force_z"], label='Fz (Full)', color='C2')
    axs2[0].plot(time, results["sindy"]["force_x"], '--', label='Fx (SINDy)', color='C0', alpha=0.7)
    axs2[0].plot(time, results["sindy"]["force_y"], '--', label='Fy (SINDy)', color='C1', alpha=0.7)
    axs2[0].plot(time, results["sindy"]["force_z"], '--', label='Fz (SINDy)', color='C2', alpha=0.7)
    axs2[0].set_title('(a) Forces vs Time')
    axs2[0].set_xlabel('Time [s]')
    axs2[0].set_ylabel('Force [N]')
    axs2[0].legend()

    # (b) Moments
    axs2[1].plot(time, results["full"]["moment_x"], label='Mx (Full)', color='C0')
    axs2[1].plot(time, results["full"]["moment_y"], label='My (Full)', color='C1')
    axs2[1].plot(time, results["full"]["moment_z"], label='Mz (Full)', color='C2')
    axs2[1].plot(time, results["sindy"]["moment_x"], '--', label='Mx (SINDy)', color='C0', alpha=0.7)
    axs2[1].plot(time, results["sindy"]["moment_y"], '--', label='My (SINDy)', color='C1', alpha=0.7)
    axs2[1].plot(time, results["sindy"]["moment_z"], '--', label='Mz (SINDy)', color='C2', alpha=0.7)
    axs2[1].set_title('(b) Moments vs Time')
    axs2[1].set_xlabel('Time [s]')
    axs2[1].set_ylabel('Moment [Nm]')
    axs2[1].legend()

    plt.tight_layout()
    plt.show()

    # ---------- FIGURE 3: 3D Trajectory Comparisons ----------
    # ---------- FIGURE 3: 3D Trajectory Comparison ----------
    fig3, ax = plt.subplots(1, 1, figsize=(10, 7), subplot_kw={'projection': '3d'})

    ax.plot(results["full"]["x"], results["full"]["y"], results["full"]["z"],
            label='Full LQR', color='C0', lw=2)
    ax.plot(results["sindy"]["x"], results["sindy"]["y"], results["sindy"]["z"],
            '--', label='SINDy LQR', color='C1', lw=2)
    ax.plot(results["dmdc"]["x"], results["dmdc"]["y"], results["dmdc"]["z"],
            '--', label='DMDC LQR', color='C3', lw=2)
    ax.plot([line_start[0], line_end[0]],
            [line_start[1], line_end[1]],
            [utils.zd, utils.zd],
            'r--', lw=2, label=f'Ref (z={utils.zd:.2f})')

    ax.set_title('3D Trajectory Comparison: Full vs SINDy vs DMDC')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.legend()
    ax.grid(True)

    # Optional: adjust camera view
    ax.view_init(elev=25, azim=-45)

    plt.tight_layout()
    plt.show()


plot_comparison(results)