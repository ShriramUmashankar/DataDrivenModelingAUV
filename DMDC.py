import numpy as np
import pandas as pd
from numpy.linalg import pinv
from sklearn.linear_model import Ridge
from finite_differences import SimpleFiniteDifference

def compute_continuous_dmdc_from_csv(
    csv_path,
    state_cols=None,
    control_cols=None,
    dt=0.1,
    remove_na=True,
    verbose=True
):
    """
    Computes continuous-time DMDc: x_dot = A x + B u from CSV.
    Uses finite differences for time derivative estimation.
    """
    # defaults
    if state_cols is None:
        state_cols = ['u','v','w','p','q','r']
    if control_cols is None:
        control_cols = ['Fx','Fy','Fz','Tx','Ty','Tz']

    # read csv
    df = pd.read_csv(csv_path)
    df = df.iloc[:4000, :]
    if remove_na:
        df = df.dropna().reset_index(drop=True)


    # Pick actual matching columns
    cols_lower = {c.lower(): c for c in df.columns}
    def pick_columns(names):
        picked = []
        for n in names:
            key = n.lower()
            if key in cols_lower:
                picked.append(cols_lower[key])
            else:
                matches = [c for c in df.columns if c.lower().replace('_','').replace(' ','') == key.replace('_','').replace(' ','')]
                if len(matches) == 1:
                    picked.append(matches[0])
                else:
                    raise KeyError(f"Column '{n}' not found. Available: {list(df.columns)}")
        return picked

    state_cols_actual = pick_columns(state_cols)
    control_cols_actual = pick_columns(control_cols)

    # Extract data
    x = df[state_cols_actual].to_numpy()
    u = df[control_cols_actual].to_numpy().T

    # Compute x_dot using finite difference
    fd = SimpleFiniteDifference(axis=0)
    x_dot = fd._differentiate(x, dt).T  # transpose so (n_state, N)

    # Transpose states too
    X = x.T  # shape (n_state, N)
    Xdot = x_dot  # shape (n_state, N)
    N = X.shape[1]
    if N < 3:
        raise ValueError("Need at least 3 timesteps.")

    # Build regression: Xdot = [A B] [X; U]
    Omega = np.vstack([X, u])  # shape (n_state+n_input, N)
    Y = Xdot                   # shape (n_state, N)


    W = Y @ pinv(Omega)

    n_state = len(state_cols_actual)
    n_input = len(control_cols_actual)
    A = W[:, :n_state]
    B = W[:, n_state:]

    # Evaluate fit
    Xdot_pred = A @ X + B @ u
    rmse = np.sqrt(np.mean((Xdot_pred - Xdot)**2))

    if verbose:
        print(f"Continuous-time DMDc RMSE: {rmse:.6g}")
        print("A shape:", A.shape, "B shape:", B.shape)

    info = {'X': X, 'U': u, 'Xdot': Xdot, 'rmse': rmse, 'dt': dt}
    return A, B, info

A, B, info = compute_continuous_dmdc_from_csv(
    "data_files/sindy_training_data.csv",
    dt=0.1
)

np.save('data_files/A.npy', A)
np.save('data_files/B.npy', B)

