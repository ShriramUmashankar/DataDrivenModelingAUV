import utils
import numpy as np
from VehicleDynamics import Dynamics
from Controller import LQRController
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 


# Assuming `dyn` is your Dynamics object
amogh = Dynamics('Amogh')
LQR = LQRController(Q_scale=6000.0, R_scale=1.0)

A = np.load('data_files/A.npy')
B = np.load('data_files/B.npy')

force_yaw = []
force_x = []
force_y = []
state_x = []
state_y = []
state_z = []
state_yaw = []
vel_x = []
vel_y = []

for i in range(utils.SIM_STEP):
    # force = np.array([80, 0, 0, 0, 0, 0]).reshape((6,1))
    # force = np.clip(force , -80, 80)
    eta = amogh.current_state[:6, :]
    v = amogh.current_state[6:, :]

    #force = LQR.compute_lqr_force(amogh.current_state, amogh.M, amogh.D @ np.diagflat(np.abs(v)), amogh.get_C(v))
    #force = LQR.compute_lqr_force_sindy(amogh.current_state)
    force = LQR.compute_lqr_force_dmdc(amogh.current_state, A, B)
    
    if utils.SIM_STEP % 100 == 0:
        print("Percentage Sim done is:", (i * 100)/utils.SIM_STEP)

    force = np.clip(force, -80, 80)

    force_yaw.append(force.flatten()[2])
    force_x.append(force.flatten()[0])
    force_y.append(force.flatten()[1])
    state_x.append(amogh.current_state.flatten()[0])
    state_y.append(amogh.current_state.flatten()[1])
    state_z.append(amogh.current_state.flatten()[2])
    state_yaw.append(amogh.current_state.flatten()[5])

    solution_matrix, solution_time = amogh.run_simulation(amogh.current_state.flatten(), force)
    vel_x.append(amogh.current_global_velocity.flatten()[0])
    vel_y.append(amogh.current_global_velocity.flatten()[1])


def plot():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'lines.linewidth': 2,
        'grid.alpha': 0.3,
        'figure.figsize': (8, 5),
    })

    # --- 1️ Force plot ---
    fig1, ax1 = plt.subplots()
    ax1.plot(force_x, label='Force X')
    ax1.plot(force_y, label='Force Y')
    ax1.plot(force_yaw, label='Force Yaw')
    ax1.set_title('Forces vs Simulation Step')
    ax1.set_xlabel('Simulation Step')
    ax1.set_ylabel('Force [N or Nm]')
    ax1.legend()
    ax1.grid(True)

    # --- 2️ XY trajectory with reference line ---
    fig2, ax2 = plt.subplots()
    ax2.plot(state_x, state_y, label='Trajectory', color='C0')

    # Reference line from utils
    line_start = np.array(utils.line_start)
    line_end = np.array(utils.line_end)

    # Construct line points for plotting
    line_points = np.vstack((line_start, line_end))
    ax2.plot(line_points[:, 0], line_points[:, 1], '--', color='r', label='Reference line')

    ax2.set_title('XY Trajectory')
    ax2.set_xlabel('X Position [m]')
    ax2.set_ylabel('Y Position [m]')
    ax2.axis('equal')
    ax2.legend()
    ax2.grid(True)

    # --- 3️ Z vs step ---
    fig3, ax3 = plt.subplots()
    ax3.plot(state_z, label='Z Position', color='C2')
    ax3.set_title('Z Position vs Simulation Step')
    ax3.set_xlabel('Simulation Step')
    ax3.set_ylabel('Z Position [m]')
    ax3.legend()
    ax3.grid(True)

    # --- 4️ Yaw vs step ---
    fig4, ax4 = plt.subplots()
    ax4.plot(np.array(state_yaw) * 180 / np.pi, label='Yaw', color='C3')
    ax4.set_title('Yaw Angle vs Simulation Step')
    ax4.set_xlabel('Simulation Step')
    ax4.set_ylabel('Yaw [deg]')
    ax4.legend()
    ax4.grid(True)

    # # --- 5️ 3D trajectory + reference line ---
    # fig5 = plt.figure(figsize=(8, 6))
    # ax5 = fig5.add_subplot(111, projection='3d')

    # ax5.plot(state_x, state_y, state_z, label='Trajectory', color='C0', lw=2)
    # zd = utils.zd
    # ax5.plot(line_points[:, 0], line_points[:, 1], [zd, zd], '--', color='r', lw=2, label='Reference line (z = {:.2f})'.format(zd))

    # ax5.set_title('3D Trajectory')
    # ax5.set_xlabel('X [m]')
    # ax5.set_ylabel('Y [m]')
    # ax5.set_zlabel('Z [m]')
    # ax5.legend()
    # ax5.grid(True)

    plt.tight_layout()
    plt.show()


plot()