import numpy as np

def J_matrix(state):
    """
    Compute full 6x6 transformation matrix for 3D AUV motion.
    Converts body-frame velocities ν = [u,v,w,p,q,r] into
    global-frame rates η̇ = [ẋ,ẏ,ż,φ̇,θ̇,ψ̇].

    state: (6,1) or (12,1) vector containing Euler angles (φ, θ, ψ)
           at indices [3,4,5].
    """
    # Extract Euler angles
    phi, theta, psi = state[3, 0], state[4, 0], state[5, 0]

    # Rotation from body frame to inertial frame (J1)
    J1 = np.array([
        [np.cos(psi)*np.cos(theta),
         np.cos(psi)*np.sin(theta)*np.sin(phi) - np.sin(psi)*np.cos(phi),
         np.cos(psi)*np.sin(theta)*np.cos(phi) + np.sin(psi)*np.sin(phi)],
        [np.sin(psi)*np.cos(theta),
         np.sin(psi)*np.sin(theta)*np.sin(phi) + np.cos(psi)*np.cos(phi),
         np.sin(psi)*np.sin(theta)*np.cos(phi) - np.cos(psi)*np.sin(phi)],
        [-np.sin(theta),
         np.cos(theta)*np.sin(phi),
         np.cos(theta)*np.cos(phi)]
    ])

    # Angular rate transformation (J2)
    J2 = np.array([
        [1, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)]
    ])

    # Assemble full 6x6 matrix
    O3 = np.zeros((3, 3))
    J = np.block([[J1, O3],
                  [O3, J2]])

    return J, J1, J2


def J_dot(psi, psi_dot):
    c, s = np.cos(psi), np.sin(psi)
    dJ = psi_dot * np.array([
        [-s, -c, 0],
        [ c, -s, 0],
        [ 0,  0, 0]
    ])
    return dJ


start_state = np.array([-40, -50, -20, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((12, 1))
desired_state = np.array([0, 0, -10, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((12, 1))

line_start = np.array([-10,-80])
line_end   = np.array([0.0, 0.0])
zd = -10

def wrap_angle(angle):
    """Wrap angle to (-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

SIM_STEP = 500

dt = 0.1