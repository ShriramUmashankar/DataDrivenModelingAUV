import numpy as np
import control as ctrl
import utils

class PIDController:
    def __init__(self, Kp, Ki, Kd, dt=0.1):
        self.Kp = np.array(Kp).reshape((3,))
        self.Ki = np.array(Ki).reshape((3,))
        self.Kd = np.array(Kd).reshape((3,))
        self.dt = dt
        self.integral_error = np.zeros((3,))
        self.prev_error = np.zeros((3,))

    def reset(self):
        self.integral_error[:] = 0
        self.prev_error[:] = 0

    def compute_force(self, current_state, desired_state):
        """
        Inputs:
            current_state: np.array (6,1) -> [x, y, psi, u, v, r]
            desired_state: np.array (6,1) -> [x_d, y_d, psi_d, 0, 0, 0]
        Outputs:
            force: np.array (3,1) -> [Fx_body, Fy_body, Mz]
        """

        # Extract states
        pos = current_state[0:3].flatten()      # [x, y, psi]
        vel = current_state[3:6].flatten()      # [u, v, r]
        pos_d = desired_state[0:3].flatten()    # [x_d, y_d, psi_d]
        vel_d = desired_state[3:6].flatten()    # desired velocities (zeros)

        # Position error in global frame
        pos_error = pos_d - pos

        # Transform position error to body frame using J^T (inverse of rotation)
        psi = pos[2]
        R = np.array([[np.cos(psi), np.sin(psi), 0],
                      [-np.sin(psi), np.cos(psi), 0],
                      [0, 0, 1]])
        error_body = R @ pos_error  # position and orientation error in body frame

        # Velocity error (assuming desired velocity = 0)
        vel_error = -vel

        # PID terms
        self.integral_error += error_body * self.dt
        derivative_error = (error_body - self.prev_error) / self.dt
        self.prev_error = error_body

        # PID output
        force = (
            self.Kp * error_body +
            self.Ki * self.integral_error +
            self.Kd * derivative_error +
            self.Kp * vel_error  # velocity error as part of damping
        )

        return force.reshape((3, 1))

class LQRController:
    def __init__(self, Q_scale, R_scale):
        self.Q_scale = Q_scale
        self.R_scale = R_scale
        self.z_obj_lqr = utils.desired_state[2, 0]  # desired z position
        self.z_vel_lqr = 0.0

        # line equation coefficients (y = m*x + c)
        self.m_lqr, self.c_lqr = self.compute_line_eq(utils.line_start, utils.line_end)

    # --------------------------------------------------------------
    def compute_line_eq(self, start, end):
        """Compute slope (m) and intercept (c) for line through start and end points"""
        x1, y1 = start[0], start[1]
        x2, y2 = end[0], end[1]
        if abs(x2 - x1) < 1e-6:
            # vertical line
            m = np.inf
            c = np.nan
        else:
            m = (y2 - y1) / (x2 - x1)
            c = y1 - m * x1
        return m, c

    # --------------------------------------------------------------
    def perpendicular_distance(self, x, y):
        """Compute perpendicular distance of (x, y) from the line y = m*x + c"""
        if np.isinf(self.m_lqr):
            # vertical line case
            distance = x - utils.line_start[0]
        else:
            distance = (y - self.m_lqr * x - self.c_lqr) / np.sqrt(1 + self.m_lqr**2)
        return distance

    # --------------------------------------------------------------
    def state_matrix(self, state, M, D, C):
        """Construct continuous-time A and B matrices"""
        J, _, _ = utils.J_matrix(state)
        O6 = np.zeros((6, 6))
        A = np.concatenate([
            np.concatenate([O6, J], axis=1),
            np.concatenate([O6, -np.linalg.inv(M) @ (C + D)], axis=1)
        ], axis=0)
        B = np.concatenate([O6, np.linalg.inv(M)], axis=0)
        return A, B

    # --------------------------------------------------------------
    def compute_lqr_force(self, current_state, M, D, C):
        """
        Compute LQR control force for line following and constant depth.
        Inputs:
            current_state: (12,1)
            M, D, C: system matrices
        Returns:
            Force (6x1)
        """

        # ------------------------------------------------------------------
        # Split state
        # ------------------------------------------------------------------
        X1 = current_state[0:6].reshape(6, 1)   # [x, y, z, roll, pitch, yaw]
        V1 = current_state[6:12].reshape(6, 1)  # [u, v, w, p, q, r]

        # ------------------------------------------------------------------
        # Desired state (for reference)
        # ------------------------------------------------------------------
        Xd = utils.desired_state[0:6].reshape(6, 1)
        Vd = utils.desired_state[6:12].reshape(6, 1)

        # ------------------------------------------------------------------
        # Compute perpendicular distance to the line (in XY plane)
        # ------------------------------------------------------------------
        x = X1[0, 0]
        y = X1[1, 0]
        distance = -self.perpendicular_distance(x, y)

        # ------------------------------------------------------------------
        # LQR setup
        # ------------------------------------------------------------------
        Q = self.Q_scale * np.eye(12)
        R = self.R_scale * np.eye(6)
        A, B = self.state_matrix(X1, M, D, C)
        K, _, _ = ctrl.lqr(A, B, Q, R)

        # ------------------------------------------------------------------
        # Build error state
        # ------------------------------------------------------------------
        e_state = np.zeros((12, 1))

        # Orientation errors
        #e_state[4, 0] = -X1[4, 0]                      # pitch correction
        #e_state[5, 0] = np.arctan(self.m_lqr) - X1[5, 0]        # maintain constant yaw (45° example)
    
        # Velocity & depth errors
        e_state[7, 0] = distance - V1[1, 0]            # lateral error and body-y velocity
        e_state[8, 0] = self.z_obj_lqr - X1[2, 0] - V1[2, 0]   # z tracking
        e_state[10, 0] = -V1[4, 0]                     # angular velocity dampers
        e_state[11, 0] = utils.wrap_angle(np.arctan(self.m_lqr) - X1[5, 0]) - V1[5, 0]

        # ------------------------------------------------------------------
        # Control computation
        # ------------------------------------------------------------------
        intermediate = K @ e_state

        Force = np.zeros((6, 1))
        Force[0:3, 0] = intermediate[0:3, 0]
        Force[3, 0] = 0
        Force[4:6, 0] = intermediate[4:6, 0]
        Force[0, 0] = 10

        return Force
    
    def compute_lqr_force_sindy(self, current_state):
        """
        Compute LQR control force for line following and constant depth.
        Inputs:
            current_state: (12,1)
            M, D, C: system matrices
        Returns:
            Force (6x1)
        """

        # ------------------------------------------------------------------
        # Split state
        # ------------------------------------------------------------------
        X1 = current_state[0:6].reshape(6, 1)   # [x, y, z, roll, pitch, yaw]
        V1 = current_state[6:12].reshape(6, 1)  # [u, v, w, p, q, r]

        # ------------------------------------------------------------------
        # Desired state (for reference)
        # ------------------------------------------------------------------
        Xd = utils.desired_state[0:6].reshape(6, 1)
        Vd = utils.desired_state[6:12].reshape(6, 1)

        # ------------------------------------------------------------------
        # Compute perpendicular distance to the line (in XY plane)
        # ------------------------------------------------------------------
        x = X1[0, 0]
        y = X1[1, 0]
        distance = -self.perpendicular_distance(x, y)

        # ------------------------------------------------------------------
        # LQR setup
        # ------------------------------------------------------------------
        Q = self.Q_scale * np.eye(6)
        R = self.R_scale * np.eye(6)
        
        A = np.array([
            [-0.445000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000],
            [ 0.000000, -0.438000,  0.000000,  0.000000,  0.000000,  0.000000],
            [ 0.000000,  0.000000, -0.430000,  0.213319,  0.000000,  0.000000],
            [ 0.000000,  0.000000, -0.254422,  0.000000,  0.000000,  0.000000],
            [ 0.000000,  0.000000,  0.000000,  0.000000, -0.458000,  0.000000],
            [-0.110507,  0.000000,  0.000000,  0.000000,  0.000000, -0.451000]
        ])

        B = np.array([
            [0.0335016,  0.0000000,  0.0000000,  0.0000000,  0.0000000,  0.0000000],
            [0.0000000,  0.0118381,  0.0000000,  0.0000000,  0.0000000,  0.0000000],
            [0.0000000,  0.0000000,  0.0106000,  0.0000000,  0.0000000,  0.0000000],
            [0.0000000,  0.0000000,  0.0000000,  0.0000000,  0.0000000,  0.0000000],
            [0.0000000,  0.0000000,  0.0000000,  0.0000000,  0.0096562,  0.0000000],
            [0.0000000,  0.0000000,  0.0000000,  0.0000000,  0.0000000,  0.0119818]
        ])

        K, _, _ = ctrl.lqr(A, B, Q, R)

        # ------------------------------------------------------------------
        # Build error state
        # ------------------------------------------------------------------
        e_state = np.zeros((6, 1))

        # Orientation errors
        #e_state[4, 0] = -X1[4, 0]                      # pitch correction
        #e_state[5, 0] = np.arctan(self.m_lqr) - X1[5, 0]        # maintain constant yaw (45° example)
    
        # Velocity & depth errors
        e_state[1, 0] = distance - V1[1, 0]            # lateral error and body-y velocity
        e_state[2, 0] = self.z_obj_lqr - X1[2, 0] - V1[2, 0]   # z tracking
        e_state[4, 0] = -V1[4, 0]                     # angular velocity dampers
        e_state[5, 0] = utils.wrap_angle(np.arctan(self.m_lqr) - X1[5, 0]) - V1[5, 0]

        # ------------------------------------------------------------------
        # Control computation
        # ------------------------------------------------------------------
        intermediate = K @ e_state

        Force = np.zeros((6, 1))
        Force[0:3, 0] = intermediate[0:3, 0]
        Force[3, 0] = 0
        Force[4:6, 0] = intermediate[4:6, 0]
        Force[0, 0] = 10

        return Force
    
    def compute_lqr_force_dmdc(self, current_state, A, B):
        """
        Compute LQR control force for line following and constant depth.
        Inputs:
            current_state: (12,1)
            M, D, C: system matrices
        Returns:
            Force (6x1)
        """

        # ------------------------------------------------------------------
        # Split state
        # ------------------------------------------------------------------
        X1 = current_state[0:6].reshape(6, 1)   # [x, y, z, roll, pitch, yaw]
        V1 = current_state[6:12].reshape(6, 1)  # [u, v, w, p, q, r]

        # ------------------------------------------------------------------
        # Desired state (for reference)
        # ------------------------------------------------------------------
        Xd = utils.desired_state[0:6].reshape(6, 1)
        Vd = utils.desired_state[6:12].reshape(6, 1)

        # ------------------------------------------------------------------
        # Compute perpendicular distance to the line (in XY plane)
        # ------------------------------------------------------------------
        x = X1[0, 0]
        y = X1[1, 0]
        distance = -self.perpendicular_distance(x, y)

        # ------------------------------------------------------------------
        # LQR setup
        # ------------------------------------------------------------------
        Q = self.Q_scale * np.eye(6)
        R = self.R_scale * np.eye(6)
        

        K, _, _ = ctrl.lqr(A, B, Q, R)

        # ------------------------------------------------------------------
        # Build error state
        # ------------------------------------------------------------------
        e_state = np.zeros((6, 1))

        # Orientation errors
        #e_state[4, 0] = -X1[4, 0]                      # pitch correction
        #e_state[5, 0] = np.arctan(self.m_lqr) - X1[5, 0]        # maintain constant yaw (45° example)
    
        # Velocity & depth errors
        e_state[1, 0] = distance - V1[1, 0]            # lateral error and body-y velocity
        e_state[2, 0] = self.z_obj_lqr - X1[2, 0] - V1[2, 0]   # z tracking
        e_state[4, 0] = -V1[4, 0]                     # angular velocity dampers
        e_state[5, 0] = utils.wrap_angle(np.arctan(self.m_lqr) - X1[5, 0]) - V1[5, 0]

        # ------------------------------------------------------------------
        # Control computation
        # ------------------------------------------------------------------
        intermediate = K @ e_state

        Force = np.zeros((6, 1))
        Force[0:3, 0] = intermediate[0:3, 0]
        Force[3, 0] = 0
        Force[4:6, 0] = intermediate[4:6, 0]
        Force[0, 0] = 10

        return Force