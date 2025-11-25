import numpy as np

class SimpleFiniteDifference:
    """
    Calculates the first derivative (d=1) using second-order (order=2) 
    finite differences on a uniform grid.

    This is a simplified version inspired by the PySINDy FiniteDifference
    class, focusing on the common use case: d=1, order=2, uniform grid.

    It uses:
    - Second-order centered differences for the interior points.
    - Second-order forward/backward differences for the endpoints.
    
    This implementation assumes the data `x` is provided as an array
    where the differentiation axis (e.g., time) is specified by `axis`.
    """

    def __init__(self, axis=0):
        """
        Parameters
        ----------
        axis : int, optional (default 0)
            The axis of the input array `x` along which to differentiate.
            For data `x` with shape (n_samples, n_features), use `axis=0`.
        """
        self.axis = axis

    def _differentiate(self, x, dt):
        """
        Compute the derivative of `x` with respect to the time step `dt`.

        Parameters
        ----------
        x : np.ndarray
            The data to differentiate.
        dt : float
            The time step (grid spacing), e.g., t[1] - t[0].

        Returns
        -------
        np.ndarray
            The computed derivative, `x_dot`, with the same shape as `x`.
        """
        
        # Ensure x is a numpy array
        x = np.asarray(x)
        
        # Create an output array of the same shape
        x_dot = np.zeros_like(x)

        # Move the differentiation axis to the first position (axis 0)
        # This simplifies the slicing logic, making it general for any axis.
        x_roll = np.moveaxis(x, self.axis, 0)
        x_dot_roll = np.moveaxis(x_dot, self.axis, 0)
        
        n_t = x_roll.shape[0]

        # Check if data is long enough
        if n_t < 3:
            raise ValueError(
                f"Data along axis {self.axis} is too short (size {n_t}). "
                "Need at least 3 points for 2nd-order differences."
            )

        # --- Interior points: 2nd-order centered difference ---
        # f'(i) = (f(i+1) - f(i-1)) / (2*dt)
        x_dot_roll[1:-1] = (x_roll[2:] - x_roll[:-2]) / (2 * dt)

        # --- Boundary points: 2nd-order forward/backward ---
        
        # Left boundary (i=0): 2nd-order forward difference
        # f'(0) = (-3*f(0) + 4*f(1) - f(2)) / (2*dt)
        x_dot_roll[0] = (-3.0 * x_roll[0] + 4.0 * x_roll[1] - x_roll[2]) / (2 * dt)
        
        # Right boundary (i=-1): 2nd-order backward difference
        # f'(-1) = (3*f(-1) - 4*f(-2) + f(-3)) / (2*dt)
        x_dot_roll[-1] = (3.0 * x_roll[-1] - 4.0 * x_roll[-2] + x_roll[-3]) / (2 * dt)

        # Move the axis back to its original position and return
        return np.moveaxis(x_dot_roll, 0, self.axis)

# --- Example Usage ---
# This code will only run if the file is executed directly.
if __name__ == "__main__":
    
    print("--- SimpleFiniteDifference (order=2) Example ---")
    
    # 1. Setup the data (mimics the PySINDy docstring)
    # 5 time points, 2 features (sin(t) and cos(t))
    t = np.linspace(0, 1, 5)
    X = np.vstack((np.sin(t), np.cos(t))).T
    
    # Calculate the uniform time step
    dt = t[1] - t[0]
    
    print(f"Input data shape (n_samples, n_features): {X.shape}")
    print(f"Time vector: {t}")
    print(f"Time step (dt): {dt}\n")
    print("Input X (features are sin(t), cos(t)):")
    print(X)

    # 2. Initialize and use the differentiator
    # We want to differentiate along axis 0 (the time sample axis)
    fd = SimpleFiniteDifference(axis=0)
    
    x_dot = fd._differentiate(X, dt)
    
    print("\nComputed derivative x_dot:")
    print(x_dot)

    # 3. Compare with the true (analytical) derivative
    # d/dt(sin(t)) = cos(t)
    # d/dt(cos(t)) = -sin(t)
    x_dot_true = np.vstack((np.cos(t), -np.sin(t))).T
    
    print("\nTrue (analytical) derivative:")
    print(x_dot_true)

    # 4. Calculate the error
    error = np.linalg.norm(x_dot - x_dot_true)
    print(f"\nTotal L2 Error: {error}")
    
    # Note: The output will be very close to the true derivative.
    # It will also match the output of the reference PySINDy class
    # if you run it with `is_uniform=True` and `d=1`, `order=2`.