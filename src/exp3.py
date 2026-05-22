from scipy.integrate import solve_ivp
import numpy as np
from dmd import DMD, EDMD, KoopmanVisualizer

def nonlinear_3d_system(t, state, sigma=-0.3, lmbda=-0.6, alpha=-0.9, gamma=1.5, delta=1.0, beta=-0.5):
    x, y, z = state
    dxdt = sigma * x
    dydt = lmbda * y + gamma * (x**2)
    dzdt = alpha * z + delta * (y**2) + beta * (x**5)
    return [dxdt, dydt, dzdt]

x0 = np.array([2.2, -1.4, 0.2])
t_span = (0, 2)  # Kept in sync with Exp 1 and 2
t_eval = np.linspace(t_span[0], t_span[1], 500)
sol = solve_ivp(nonlinear_3d_system, t_span, y0=x0, t_eval=t_eval, method='RK45', rtol=1e-12, atol=1e-12)

true_trajectory = sol.y
snapshots_X = true_trajectory[:, :-1]
snapshots_Y = true_trajectory[:, 1:]
n_prediction_steps = true_trajectory.shape[1]

def observables_expanded(state):
    x, y, z = state
    return np.array([x, y, z, x**2, y**2, x**5, (x**2)*y, x**4, x**3])

dmd = DMD(energy_threshold=0.9999)
dmd.fit(true_trajectory)
dmd_predictions = dmd.predict(n_steps=n_prediction_steps)

edmd = EDMD(observables_fn=observables_expanded)
edmd.fit(snapshots_X, snapshots_Y)
edmd_predictions = edmd.predict(snapshots_X, n_steps=n_prediction_steps)

print(f"=== EXPERIMENT 3: NONLINEAR COMPARISON ===")
print(f"Classical DMD Global Frobenius Error: {np.linalg.norm(true_trajectory - dmd_predictions):.4f}")
print(f"Extended EDMD Global Frobenius Error: {np.linalg.norm(true_trajectory - edmd_predictions):.4e}")