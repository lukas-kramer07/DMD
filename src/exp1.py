from scipy.integrate import solve_ivp
import numpy as np
from dmd import DMD, KoopmanVisualizer

def linear_system(t, state, sigma=-0.3, lmbda=-0.6, alpha=-0.9):
    x, y, z = state
    return [sigma * x, lmbda * y, alpha * z]

x0 = np.array([1.2, -0.4, 0.2])
# Shorten window to capture active dynamics and increase snapshot density
t_span = (0, 2)
t_eval = np.linspace(t_span[0], t_span[1], 500) 
sol = solve_ivp(linear_system, t_span, y0=x0, t_eval=t_eval, method='RK45', rtol=1e-12, atol=1e-12)

true_trajectory = sol.y
n_prediction_steps = true_trajectory.shape[1]

dmd = DMD(energy_threshold=1) # Force full numeric expansion
dmd.fit(true_trajectory)
dmd_predictions = dmd.predict(n_steps=n_prediction_steps)

print(f"=== EXPERIMENT 1: LINEAR BASELINE ===")
print(f"DMD Global Frobenius Error: {np.linalg.norm(sol.y - dmd_predictions):.4e}")