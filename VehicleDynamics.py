import numpy as np
from scipy.integrate import solve_ivp
import yaml
import utils


class Dynamics:
    """
    6-DOF AUV Dynamics Model
    State vector y = [η; ν] = [x, y, z, φ, θ, ψ, u, v, w, p, q, r]
    η: position/orientation (global frame)
    ν: linear/angular velocity (body frame)
    """

    def __init__(self, name, param_path="/home/shriram/Documents/Amogh/SimulationEnvironment/v4/parameters.yaml"):
        self.name = name
        self.path = param_path
        self.load_parameters(param_path)

        # initial conditions
        self.current_state = utils.start_state  # shape (12,1)
        self.current_acceleration = np.zeros((6, 1))
        self.current_global_velocity = np.zeros((6, 1))

        self.time_step = 0.1  # integration step

    # ----------------------------------------------------------------------
    def load_parameters(self, yaml_file):
        """Load vehicle parameters from YAML file"""
        with open(yaml_file, 'r') as file:
            params = yaml.safe_load(file)

        # physical parameters
        self.m = params["parameters"]["mass"]
        self.b = params["parameters"]["buoyancy"]

        self.Ix = params["parameters"]["Ix"]
        self.Iy = params["parameters"]["Iy"]
        self.Iz = params["parameters"]["Iz"]

        self.xg = params["parameters"]["xg"]
        self.yg = params["parameters"]["yg"]
        self.zg = params["parameters"]["zg"]

        self.xb = params["parameters"]["xb"]
        self.yb = params["parameters"]["yb"]
        self.zb = params["parameters"]["zb"]

        # matrices
        self.Mass = np.array(params["mass_matrix"])
        self.Added_mass = np.array(params["added_mass_matrix"])
        self.D = np.array(params["damping_matrix"])
        self.M = self.Mass + self.Added_mass
        self.thrust_allocation_matrix = np.array(params["thrust_allocation_matrix"])

    # ----------------------------------------------------------------------
    def get_C(self, v):
        """Coriolis and centripetal matrix for 6-DOF body velocities"""
        u, v_, w, p, q, r = v.flatten()
        m = self.m
        xg, yg, zg = self.xg, self.yg, self.zg
        Ix, Iy, Iz = self.Ix, self.Iy, self.Iz

        C = np.array([
            [0, 0, 0, m*(zg*q - yg*r), m*(w - xg*q), -m*(v_ + xg*r)],
            [0, 0, 0, -m*(w - xg*q), m*(zg*r + xg*p), m*u],
            [0, 0, 0, -m*(zg*p - v_), -m*(zg*q + u), m*(xg*p + yg*q)],
            [-m*(zg*q - yg*r), m*(w - xg*q), m*(zg*p - v_), 0, Iz*r, -Iy*q],
            [-m*(w - xg*q), -m*(zg*r + xg*p), m*(zg*q + u), -Iz*r, 0, Ix*p],
            [m*(v_ + xg*r), -m*u, -m*(xg*p + yg*q), Iy*q, -Ix*p, 0]
        ])
        return C

    # ----------------------------------------------------------------------
    def get_G(self, eta):
        """Restoring (gravity and buoyancy) forces and moments"""
        m, b = self.m, self.b
        xg, yg, zg = self.xg, self.yg, self.zg
        xb, yb, zb = self.xb, self.yb, self.zb
        phi, theta = eta[3, 0], eta[4, 0]

        G = np.array([
            [(m - b) * np.sin(theta)],
            [-(m - b) * np.cos(theta) * np.sin(phi)],
            [-(m - b) * np.cos(theta) * np.cos(phi)],
            [-(yg*m - yb*b) * np.cos(theta) * np.cos(phi) + (zg*m - zb*b) * np.cos(theta) * np.sin(phi)],
            [(zg*m - zb*b) * np.sin(theta) + (xg*m - xb*b) * np.cos(theta) * np.cos(phi)],
            [-(xg*m - xb*b) * np.cos(theta) * np.sin(phi) - (yg*m - yb*b) * np.sin(theta)]
        ])
        return G

    # ----------------------------------------------------------------------
    def solver(self, t, y, tau):
        """Compute the time derivative dy/dt = f(y, tau)"""
        eta = y[:6].reshape((6, 1))  # global position/orientation
        nu = y[6:].reshape((6, 1))   # body velocity

        # quadratic damping term |v|v
        nu_abs = np.abs(nu)
        Dv = self.D @ (nu * nu_abs)

        # Coriolis and restoring forces
        C = self.get_C(nu)
        G = self.get_G(eta)

        # rigid-body + added mass inverse
        Minv = np.linalg.inv(self.M)

        # linear + angular acceleration
        nudot = Minv @ (tau - C @ nu - Dv - G)
        self.current_acceleration = nudot

        # global kinematics (η̇ = J(η) ν)
        J, _, _ = utils.J_matrix(eta)
        etadot = J @ nu
        self.current_global_velocity = etadot

        return np.vstack((etadot, nudot)).flatten()

    # ----------------------------------------------------------------------
    def run_simulation(self, initial_state, tau):
        """Run simulation for one integration step"""
        t_span = (0, self.time_step)
        sol = solve_ivp(self.solver, t_span, initial_state.flatten(), args=(tau,), t_eval=[self.time_step])

        self.current_state = sol.y[:, -1].reshape((12, 1))
        return sol.y, sol.t
