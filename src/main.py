from scipy.integrate import solve_ivp
import numpy as np
from dmd import DMD, EDMD, KoopmanVisualizer

# 1. Define a 3D cascading nonlinear system with a finite Koopman subspace
def finite_3d_koopman_system(t, state, sigma=-0.3, lmbda=-0.6, alpha=-0.9, gamma=1.5, delta=1.0, beta=-0.5):
    """
    3D coupled nonlinear system. Possesses an exact, 9-dimensional 
    closed Koopman invariant subspace despite mixed 2nd and 5th order terms.
    """
    x, y, z = state
    dxdt = sigma * x
    dydt = lmbda * y + gamma * (x**2)
    dzdt = alpha * z + delta * (y**2) + beta * (x**5)
    return [dxdt, dydt, dzdt]

# 2. Generate clean continuous simulation snapshots
x0 = np.array([10.2, -0.4, 0.2])
t_span = (0, 8)
t_eval = np.linspace(t_span[0], t_span[1], 250)
sol = solve_ivp(finite_3d_koopman_system, t_span, y0=x0, t_eval=t_eval, method='RK45')

# Unpack baseline 3D data arrays
true_trajectory = sol.y
snapshots_X = true_trajectory[:, :-1]
snapshots_Y = true_trajectory[:, 1:]
n_prediction_steps = true_trajectory.shape[1]
state_names = ['x', 'y', 'z']

# 3. Define the expanded closed dictionary matching the coupled system dynamics
def observables_expanded(state):
    """
    Strictly closed 9-dimensional Koopman dictionary.
    Contains all interacting nonlinear terms required for exact algebraic closure.
    """
    x, y, z = state
    return np.array([
        x, 
        y, 
        z, 
        x**2, 
        y**2, 
        x**5,
        (x**2) * y,
        x**4,
        x**3
    ])

# 4. Instantiate and evaluate the linear DMD Model
dmd = DMD(energy_threshold=0.9999)
dmd.fit(true_trajectory)
dmd_predictions = dmd.predict(n_steps=n_prediction_steps)

# 5. Instantiate and evaluate the non-linear EDMD Model (Pure, Unregularized)
edmd = EDMD(observables_fn=observables_expanded)
edmd.fit(snapshots_X, snapshots_Y)
edmd_predictions = edmd.predict(snapshots_X, n_steps=n_prediction_steps)

# 6. Generate Verification Diagnostics via Visualizer
viz = KoopmanVisualizer()

# Plot trajectory tracking performance across all 3 states
viz.plot_predictions(sol.t, true_trajectory, dmd_predictions, labels=state_names, title="DMD Predictions (3D)")
viz.plot_predictions(sol.t, true_trajectory, edmd_predictions, labels=state_names, title="EDMD Predictions (3D)")

# Plot logarithmic error curves over the simulation horizon
viz.plot_error_growth(true_trajectory, dmd_predictions, title="DMD 3D Error Trajectory")
viz.plot_error_growth(true_trajectory, edmd_predictions, title="EDMD 3D Error Trajectory")

# Plot the singular values spectrum profiles
viz.plot_singular_values(dmd, title="DMD Data Matrix Singular Values (3D)")
viz.plot_singular_values(edmd, title="EDMD Gramian Singular Values (3D)")

# Extract and view the discrete eigenvalue spectrum layout inside the unit circle
viz.plot_eigenvalues(dmd.Lambda, title="DMD Discrete Operator Spectrum")
viz.plot_eigenvalues(edmd.mu, title="EDMD Koopman Operator Spectrum")

# Reconstruct and render the modes matrix V heatmap configuration
viz.visualize_modes(xi=edmd.xi, n_states=3, k_observables=9, title="EDMD Extracted Modes Matrix (V)")

# Print error comparison metrics
print(f"DMD Global Frobenius Error: {np.linalg.norm(sol.y - dmd_predictions):.4f}")
print(f"EDMD Rich Global Frobenius Error: {np.linalg.norm(sol.y - edmd_predictions):.4e}")