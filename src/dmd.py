import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd

class DMD:
    def __init__(self, energy_threshold=0.999):
        """
        Classical Dynamic Mode Decomposition with energy-based truncation.
        """
        self.energy_threshold = energy_threshold
        self.Phi = None        # Koopman / DMD Modes
        self.Lambda = None     # Discrete-time Eigenvalues
        self.b = None          # Mode Amplitudes
        self.Sigma = None      # Singular values of the data matrix X

    def fit(self, snapshots, target_rank=None):
        """
        Fits the DMD operator with optional hard-rank subspace truncation.
        """
        X = snapshots[:, :-1]
        X_prime = snapshots[:, 1:]
        
        U, Sigma, Vt = svd(X, full_matrices=False)
        self.Sigma = Sigma
        
        # Determine truncation rank
        if target_rank is not None:
            r = min(target_rank, len(Sigma))
        else:
            energy = Sigma**2
            cumulative_energy = np.cumsum(energy) / np.sum(energy)
            r = np.searchsorted(cumulative_energy, self.energy_threshold) + 1
        
        print(f"[Diagnostic] Fitted DMD with truncation rank r = {r}")
        
        U_r = U[:, :r]
        Sigma_r = Sigma[:r]
        Vt_r = Vt[:r, :]
        
        A_tilde = U_r.T @ X_prime @ Vt_r.T @ np.linalg.inv(np.diag(Sigma_r))
        
        self.Lambda, W = np.linalg.eig(A_tilde)
        self.Phi = X_prime @ Vt_r.T @ np.linalg.inv(np.diag(Sigma_r)) @ W
        
        x0 = snapshots[:, 0:1]
        b_vec, _, _, _ = np.linalg.lstsq(self.Phi, x0, rcond=None)
        self.b = b_vec.flatten()

    def predict(self, n_steps):
        """
        Propagates the linear state forward for n_steps.
        Output shape: (n_states, n_steps)
        """
        n_states = self.Phi.shape[0]
        x_pred = np.zeros((n_states, n_steps))
        
        for k in range(n_steps):
            # Evaluate time-evolution equation: Phi @ (Lambda^k * b)
            x_k = self.Phi @ (self.Lambda**k * self.b)
            x_pred[:, k] = np.real(x_k)
            
        return x_pred


class EDMD:
    def __init__(self, observables_fn):
        """
        Extended Dynamic Mode Decomposition (Row-Convention).
        """
        self.observables_fn = observables_fn
        self.K_hat = None   # Finite-dimensional Koopman Operator Matrix
        self.mu = None      # Koopman Eigenvalues
        self.xi = None      # Right Eigenvectors of K_hat
        self.G_mat = None   # Gramian Matrix of X
        self.Sigma = None   # Singular values of Gramian Matrix G

    def fit(self, snapshots_X, snapshots_Y):
        """
        Fits the EDMD operator using shifted snapshot pairings.
        Shapes: (n_states, n_snapshots)
        """
        M = snapshots_X.shape[1]
        
        # 1. Lift raw states into the high-dimensional dictionary space
        psi_X = np.array([self.observables_fn(snapshots_X[:, m]) for m in range(M)])  # (M, K)
        psi_Y = np.array([self.observables_fn(snapshots_Y[:, m]) for m in range(M)])  # (M, K)
        
        # 2. Compute Gramian and cross-correlation matrices
        self.G_mat = 1.0 / M * psi_X.T.conj() @ psi_X  # (K, K)
        A_mat = 1.0 / M * psi_X.T.conj() @ psi_Y       # (K, K)
        
        # 3. Robust pseudoinverse of G via SVD truncation
        U_g, S_g, Vt_g = np.linalg.svd(self.G_mat, full_matrices=False)
        self.Sigma = S_g
        
        threshold = 1e-10
        rank = np.sum(S_g > threshold)
        S_g_inv = np.zeros_like(S_g)
        S_g_inv[:rank] = 1.0 / S_g[:rank]
        
        G_pinv = Vt_g.T[:, :rank] @ np.diag(S_g_inv[:rank]) @ U_g[:, :rank].T
        
        # 4. Extract Koopman Matrix and execute Eigendecomposition
        self.K_hat = G_pinv @ A_mat
        self.mu, self.xi = np.linalg.eig(self.K_hat)

    def predict(self, snapshots_X, n_steps):
        """
        Propagates nonlinear dynamics via Koopman eigenfunctions.
        Output shape: (n_states, n_steps)
        """
        x_0 = snapshots_X[:, 0]
        psi_0 = self.observables_fn(x_0)
        
        # Initial evaluation of eigenfunctions matching the script convention
        phi_0 = psi_0 @ self.xi
        K_obs = len(psi_0)
        N_states = len(x_0)

        # State extraction matrix mapping out the first N variables
        B = np.zeros((N_states, K_obs))
        for i in range(N_states):
            B[i, i] = 1.0
        B = B.T  # (K_obs x N_states)
        
        # Reconstruct Koopman modes matrix V from right eigenvectors and B
        V = (np.linalg.inv(self.xi) @ B).T  # (N_states x K_obs)
        
        x_pred = np.zeros((N_states, n_steps))
        for k in range(n_steps):
            # Linearly propagate the eigenfunctions element-wise, then project down
            x_k = V @ (phi_0 * (self.mu**k))
            x_pred[:, k] = np.real(x_k)
            
        return x_pred


class KoopmanVisualizer:
    @staticmethod
    def plot_predictions(t_vector, true_data, pred_data, labels=None, title="Model Prediction"):
        """
        Plots true trajectories against model predictions across all states.
        Renders interactively without saving files.
        """
        n_states = true_data.shape[0]
        fig, axes = plt.subplots(n_states, 1, figsize=(11, 2.5 * n_states), sharex=True)
        if n_states == 1:
            axes = [axes]
            
        if labels is None:
            labels = [f"State {i+1}" for i in range(n_states)]
            
        for i in range(n_states):
            axes[i].plot(t_vector, true_data[i, :len(t_vector)], 'b-', label='True Data', linewidth=1.5)
            axes[i].plot(t_vector, pred_data[i, :len(t_vector)], 'r--', label='Predicted', linewidth=1.5)
            axes[i].set_ylabel(labels[i])
            axes[i].legend(loc='upper right')
            axes[i].grid(True, alpha=0.3)
            
        axes[-1].set_xlabel("Time")
        fig.suptitle(title, fontsize=12)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_error_growth(true_data, pred_data, title="Prediction Error Over Time"):
        """
        Computes and plots the log-scale Euclidean error trajectory over discrete steps.
        """
        steps = min(true_data.shape[1], pred_data.shape[1])
        errors = np.linalg.norm(true_data[:, :steps] - pred_data[:, :steps], axis=0)
        
        plt.figure(figsize=(8, 4))
        plt.semilogy(range(steps), errors, 'g-', linewidth=2, label='Euclidean Error')
        plt.xlabel('Discrete Step')
        plt.ylabel('Error Matrix Norm (Log Scale)')
        plt.title(title)
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_eigenvalues(eigenvalues, title="Discrete Eigenvalue Spectrum"):
        """
        Plots eigenvalues on the complex plane relative to the unit circle.
        Uses raw strings to prevent syntax escape sequence warnings.
        """
        plt.figure(figsize=(6, 6))
        
        # Plot Unit Circle boundaries
        theta = np.linspace(0, 2 * np.pi, 200)
        plt.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.5, label='Unit Circle')
        
        # Scatter complex eigenvalues
        plt.scatter(np.real(eigenvalues), np.imag(eigenvalues), c='crimson', marker='o', s=50, zorder=3, label=r'$\mu$')
        
        plt.axhline(0, color='black', alpha=0.3, linestyle=':')
        plt.axvline(0, color='black', alpha=0.3, linestyle=':')
        plt.xlabel(r'$\mathbb{R}e(\mu)$')
        plt.ylabel(r'$\mathbb{I}m(\mu)$')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def visualize_modes(xi, n_states, k_observables, title="Extracted Koopman Modes Matrix (V)"):
        """
        Reconstructs the precise working modes matrix V = (inv(xi) @ B).T 
        and visualizes component magnitudes as an interactive heatmap.
        """
        # 1. Initialize B matching the row-convention state extraction layout (K_obs x N_states)
        B = np.zeros((k_observables, n_states))
        for i in range(n_states):
            B[i, i] = 1.0
            
        # 2. Compute V using the correct inner dimensions: (9x9) @ (9x3) = (9x3) -> Transposed to (3x9)
        V = (np.linalg.inv(xi) @ B).T
        V_magnitudes = np.abs(V)
        
        # 3. Render Heatmap Visuals
        plt.figure(figsize=(9, 4.5))
        im = plt.imshow(V_magnitudes, cmap='YlGnBu', aspect='auto')
        plt.colorbar(im, label='Magnitude $|V_{ij}|$')
        
        state_labels = [f"State {i+1}" for i in range(n_states)]
        mode_labels = [f"Mode {j+1}" for j in range(k_observables)]
        
        plt.yticks(range(n_states), state_labels)
        plt.xticks(range(k_observables), mode_labels, rotation=45)
        plt.xlabel("Koopman Eigenfunctions ($\phi_j$)")
        plt.ylabel("Physical State Space Dimensions")
        plt.title(title)
        
        # Overlay numerical value flags onto matrix coordinates safely
        for i in range(V_magnitudes.shape[0]):
            for j in range(V_magnitudes.shape[1]):
                plt.text(j, i, f"{V_magnitudes[i, j]:.2f}", ha="center", va="center", 
                         color="black" if V_magnitudes[i, j] < 0.7 * np.max(V_magnitudes) else "white")
                         
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_singular_values(model, title="Singular Value Spectrum Decay"):
        """
        Plots the raw singular values to diagnose information density/rank decay.
        """
        if model.Sigma is None:
            print("Error: Model has not been fitted yet. No singular values found.")
            return
            
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(model.Sigma) + 1), model.Sigma, 'bo-', linewidth=1.5, markersize=4)
        plt.yscale('log')
        plt.xlabel('Index')
        plt.ylabel('Singular Value Magnitude (Log Scale)')
        plt.title(title)
        plt.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        plt.show()