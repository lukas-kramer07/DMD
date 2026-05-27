# src/exp1.py
# run from root with python3 -m src.exp1

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from src.dmd import DMD, KoopmanVisualizer

def linear_system(t, state, omega=3.5, gamma=3):
    x, y, z = state
    # x' =  omega * y  -> oscillation
    # y' = -omega * x  -> oscillation
    # z' = -gamma * z  -> exponential decay
    return [omega * y, -omega * x, -gamma * z]

# 1. params & setup 
x0 = np.array([1.0, 0.0, 5.0])
t_span = (0, 5)
t_eval = np.linspace(t_span[0], t_span[1], 50) 

# 2. numerical integration
sol = solve_ivp(linear_system, t_span, y0=x0, t_eval=t_eval, method='RK45', rtol=1e-12, atol=1e-12)
true_trajectory = sol.y
n_prediction_steps = true_trajectory.shape[1]

# 3. dmd fit & prediction
dmd = DMD(energy_threshold=1.0) 
dmd.fit(true_trajectory)
dmd_predictions = dmd.predict(n_steps=n_prediction_steps)

# 4. results
print(f"=== EXPERIMENT 1: LINEAR BASELINE ===")
print(f"DMD Global Frobenius Error: {np.linalg.norm(true_trajectory - dmd_predictions):.4e}")
print(f"Eigenvalues: {dmd.Lambda}")
print(f"Modes: {dmd.Phi}")

# 5. visualization
state_labels = ["State x", "State y", "State z"]

KoopmanVisualizer.plot_predictions(
    t_vector=t_eval, 
    true_data=true_trajectory, 
    pred_data=dmd_predictions, 
    labels=state_labels, 
    title=""
)

KoopmanVisualizer.plot_error_growth(
    true_data=true_trajectory, 
    pred_data=dmd_predictions, 
    title=""
)

KoopmanVisualizer.plot_eigenvalues(
    eigenvalues=dmd.Lambda, 
    title=""
)
