# src/exp2.py
# run from root with python3 -m src.exp2


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from src.dmd import DMD, KoopmanVisualizer

def linear_system_exp1(t, state, omega=3.5, gamma=3):
    x, y, z = state
    return [omega * y, -omega * x, -gamma * z]

# 1. params
x0 = np.array([1.0, 0.0, 5.0])
t_span = (0, 5)
t_eval = np.linspace(t_span[0], t_span[1], 100)

# 2. numerical integration
sol = solve_ivp(linear_system_exp1, t_span, y0=x0, t_eval=t_eval, method='RK45', rtol=1e-12, atol=1e-12)
clean_trajectory = sol.y
n_steps = clean_trajectory.shape[1]

# 3. embed into high dimensional state and add light noise
extra_dims = 17
total_dims = 3 + extra_dims
noise_level = 0.02  
high_dim_clean = np.vstack([clean_trajectory, np.zeros((extra_dims, n_steps))])
np.random.seed(42)
high_dim_noisy = high_dim_clean + np.random.normal(0, noise_level, high_dim_clean.shape)

# 4. fit dmd models
dmd_full = DMD()
dmd_full.fit(high_dim_noisy, target_rank=total_dims)
pred_high_full = dmd_full.predict(n_steps=n_steps)

dmd_trunc = DMD()
dmd_trunc.fit(high_dim_noisy, target_rank=3)
pred_high_trunc = dmd_trunc.predict(n_steps=n_steps)

# 5. extract physical dimensions for evaluation
physical_pred_full = np.real(pred_high_full[:3, :])
physical_pred_trunc = np.real(pred_high_trunc[:3, :])

error_full = np.linalg.norm(clean_trajectory - physical_pred_full, 'fro')
error_trunc = np.linalg.norm(clean_trajectory - physical_pred_trunc, 'fro')

# 6. results
print(f"\n=== EXPERIMENT 2: HIGH-DIMENSIONAL NOISE TRUNCATION ===")
print(f"System compressed from {total_dims} dimensions down to 3.")
print(f"Full Rank DMD (r={total_dims}) Frobenius Error: {error_full:.4f}")
print(f"Truncated SVD (r=3) Frobenius Error:   {error_trunc:.4f}")
print(f"Sigma: {dmd_trunc.Sigma}")

# Eigendecomp costs O(r³)
complexity_full = total_dims**3
complexity_trunc = 3**3

savings_pct = (1 - complexity_trunc / complexity_full) * 100
print(f"Computational complexity saved: {savings_pct:.1f}%")

# 7. visualization
state_labels = ["State x", "State y", "State z"]

KoopmanVisualizer.plot_predictions(
    t_vector=t_eval,
    true_data=clean_trajectory,
    pred_data=physical_pred_trunc,
    labels=state_labels,
    title=f"DMD (r=3) Reconstruction | $\\sigma={noise_level}$"
)

KoopmanVisualizer.plot_singular_values(
    S=dmd_full.Sigma, 
    r_trunc=3, 
    title=""
)
