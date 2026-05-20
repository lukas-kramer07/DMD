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

    def fit(self, snapshots):
        """
        Fits the DMD operator using standard snapshot matrices.
        Input shape: (n_states, n_snapshots)
        """
        X = snapshots[:, :-1]
        X_prime = snapshots[:, 1:]
        
        # 1. Singular Value Decomposition
        U, Sigma, Vt = svd(X, full_matrices=False)
        self.Sigma = Sigma
        
        # 2. Rank truncation based on cumulative energy
        energy = Sigma**2
        cumulative_energy = np.cumsum(energy) / np.sum(energy)
        r = np.searchsorted(cumulative_energy, self.energy_threshold) + 1
        
        U_r = U[:, :r]
        Sigma_r = Sigma[:r]
        Vt_r = Vt[:r, :]
        
        # 3. Compute low-dimensional companion matrix A_tilde
        A_tilde = U_r.T @ X_prime @ Vt_r.T @ np.linalg.inv(np.diag(Sigma_r))
        
        # 4. Eigendecomposition and exact mode extraction
        self.Lambda, W = np.linalg.eig(A_tilde)
        self.Phi = X_prime @ Vt_r.T @ np.linalg.inv(np.diag(Sigma_r)) @ W
        
        # 5. Compute initial amplitudes via least-squares matching x0
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
        """
        n_states = true_data.shape[0]
        fig, axes = plt.subplots(n_states, 1, figsize=(11, 3 * n_states), sharex=True)
        if n_states == 1:
            axes = [axes]
            
        if labels is None:
            labels = [f"State {i+1}" for i in range(n_states)]
            
        for i in range(n_states):
            axes[i].plot(t_vector, true_data[i, :len(t_vector)], 'o-', label='True Data', markersize=3, alpha=0.7)
            axes[i].plot(t_vector, pred_data[i, :len(t_vector)], 's--', label='Predicted', markersize=3, alpha=0.7)
            axes[i].set_ylabel(labels[i])
            axes[i].legend(loc='upper right')
            axes[i].grid(True, alpha=0.3)
            
        axes[-1].set_xlabel("Time")
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{title.lower().replace(' ', '_')}.png")
        plt.close()

    @staticmethod
    def plot_error_growth(true_data, pred_data, title="Prediction Error Over Time"):
        """
        Computes and plots the log-scale Euclidean error trajectory.
        """
        steps = min(true_data.shape[1], pred_data.shape[1])
        errors = np.linalg.norm(true_data[:, :steps] - pred_data[:, :steps], axis=0)
        
        plt.figure(figsize=(8, 4))
        plt.semilogy(range(steps), errors, 'r-', linewidth=2, label='Euclidean Error')
        plt.xlabel('Discrete Step')
        plt.ylabel('Error Matrix Norm (Log Scale)')
        plt.title(title)
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{title.lower().replace(' ', '_')}.png")
        plt.close()

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
        plt.savefig(f"{title.lower().replace(' ', '_')}.png")
        plt.close()