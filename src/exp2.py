from scipy.integrate import solve_ivp
import numpy as np
from dmd import DMD

def linear_system(t, state, sigma=-0.3, lmbda=-0.6, alpha=-0.9):
    x, y, z = state
    return [sigma * x, lmbda * y, alpha * z]

# 1. Generate the true 3D physical baseline
x0 = np.array([1.2, -0.4, 0.2])
t_span = (0, 2)
t_eval = np.linspace(t_span[0], t_span[1], 500)
sol = solve_ivp(linear_system, t_span, y0=x0, t_eval=t_eval, method='RK45', rtol=1e-12, atol=1e-12)
clean_trajectory = sol.y  # Shape: (3, 500)

# 2. Project the 3D system up to a 10-dimensional observation space
np.random.seed(42)
C = np.random.randn(10, 3)  # Dense coupling projection matrix

# High-dimensional observation space without noise
high_dim_clean = C @ clean_trajectory  # Shape: (10, 500)

# Inject measurement noise across the entire high-dimensional space
noise_level = 0.1
high_dim_noisy = high_dim_clean + np.random.normal(0, noise_level, high_dim_clean.shape)

# 3. Model A: Full Rank Evaluation (r=10, overfits to the noise dimensions)
dmd_full = DMD()
dmd_full.fit(high_dim_noisy, target_rank=10)

X_noisy_current = high_dim_noisy[:, :-1]
b_f, _, _, _ = np.linalg.lstsq(dmd_full.Phi, X_noisy_current, rcond=None)
pred_high_full = dmd_full.Phi @ (np.diag(dmd_full.Lambda) @ b_f)

# 4. Model B: Truncated SVD Evaluation (r=3, isolates the true underlying manifold)
dmd_trunc = DMD()
dmd_trunc.fit(high_dim_noisy, target_rank=3)

b_t, _, _, _ = np.linalg.lstsq(dmd_trunc.Phi, X_noisy_current, rcond=None)
pred_high_trunc = dmd_trunc.Phi @ (np.diag(dmd_trunc.Lambda) @ b_t)

# 5. Project the predictions back to the physical 3D space to evaluate tracking accuracy
# We use the pseudoinverse of C to map from 10D back to our 3D states
C_pinv = np.linalg.pinv(C)

clean_X_prime = clean_trajectory[:, 1:]
physical_pred_full = C_pinv @ np.real(pred_high_full)
physical_pred_trunc = C_pinv @ np.real(pred_high_trunc)

error_full = np.linalg.norm(clean_X_prime - physical_pred_full, 'fro')
error_trunc = np.linalg.norm(clean_X_prime - physical_pred_trunc, 'fro')

print(f"\n=== EXPERIMENT 2: HIGH-DIMENSIONAL SUBSPACE FILTERING ===")
print(f"Data Matrix Layout Shape: {high_dim_noisy.shape}")
print(f"Full Rank DMD (r=10) Mapping Error: {error_full:.4f}")
print(f"Truncated SVD (r=3) Mapping Error: {error_trunc:.4f}")