# src/exp3.py
# run from root with python3 -m src.exp3

from scipy.integrate import solve_ivp
import numpy as np
from src.dmd import DMD, EDMD, KoopmanVisualizer
OPEN = True # controls whether experiment is run on an open or closed system/set of observables

def nonlinear_2d_system(t, state, mu=-0.3, lmbda=-1.0, gamma=2.5):
    """
    x acts as the linear anchor.
    y is heavily driven by a quintic non-linearity.
    """
    x, y = state
    dxdt = mu * x**(1/4) if OPEN else mu*x
    dydt = lmbda * y + gamma * (x**5)
    return [dxdt, dydt]

def observables_expanded(state):
    x, y = state
    return np.array([x, y, x**5, x**(1/4)]) if OPEN else np.array([x,y,x**5])

# 1. params & setup
x0 = np.array([2.0, -2.0])
t_span = (0, 5)
t_eval = np.linspace(t_span[0], t_span[1], 50)

# 2. numerical integration
sol = solve_ivp(nonlinear_2d_system, t_span, y0=x0, t_eval=t_eval, method='RK45', rtol=1e-12, atol=1e-12)
true_trajectory = sol.y
snapshots_X = true_trajectory[:, :-1]
snapshots_Y = true_trajectory[:, 1:]
n_prediction_steps = true_trajectory.shape[1]

# 3. fit models
dmd = DMD(energy_threshold=1)
dmd.fit(true_trajectory)
dmd_predictions = dmd.predict(n_steps=n_prediction_steps)

edmd = EDMD(observables_fn=observables_expanded)
edmd.fit(snapshots_X, snapshots_Y)
edmd_predictions = edmd.predict(snapshots_X, n_steps=n_prediction_steps)

# 4. results
print(f"=== EXPERIMENT 3: 2D NONLINEAR COMPARISON ===")
print(f"Classical DMD Global Frobenius Error: {np.linalg.norm(true_trajectory - dmd_predictions):.4e}")
print(f"Extended EDMD Global Frobenius Error: {np.linalg.norm(true_trajectory - edmd_predictions):.4e}")

# 5. visualization
state_labels = ["State x", "State y"]

KoopmanVisualizer.plot_predictions(
    t_vector=t_eval, 
    true_data=true_trajectory, 
    pred_data=dmd_predictions, 
    edmd_data=edmd_predictions,
    labels=state_labels, 
)