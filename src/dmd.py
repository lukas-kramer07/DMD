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
    def plot_predictions(t_vector, true_data, pred_data, edmd_data=None, labels=None, title=None):
        """
        Optimized for IEEE single-column width (approx 3.5 inches).
        Can dynamically overlay an optional EDMD prediction line with high-contrast overlap.
        """
        n_states = true_data.shape[0]
        fig, axes = plt.subplots(n_states, 1, figsize=(7, 2.3 * n_states), sharex=True)
        if n_states == 1:
            axes = [axes]
            
        if labels is None:
            labels = [f"State {i+1}" for i in range(n_states)]
            
        line_true = None
        line_pred = None
        line_edmd = None
        
        for i in range(n_states):
            line_true, = axes[i].plot(t_vector, true_data[i, :len(t_vector)], 
                                      color='black', linestyle='-', linewidth=3.5, alpha=0.25)
            
            line_pred, = axes[i].plot(t_vector, pred_data[i, :len(t_vector)], 
                                      color='red', linestyle='--', linewidth=1.2)
            
            if edmd_data is not None:
                line_edmd, = axes[i].plot(t_vector, edmd_data[i, :len(t_vector)], 
                                          color='blue', linestyle='-.', linewidth=1.5)
                
            axes[i].set_ylabel(labels[i], fontsize=9)
            axes[i].grid(True, alpha=0.3)
            axes[i].tick_params(labelsize=8)
            
        axes[-1].set_xlabel("Time (s)", fontsize=9)
        
        legend_lines = [line_true, line_pred]
        legend_labels = ['True', 'Exact DMD']
        if edmd_data is not None:
            legend_lines.append(line_edmd)
            legend_labels.append('Extended DMD')
        
        fig.legend(legend_lines, legend_labels, 
                   loc='lower center', ncol=len(legend_lines), bbox_to_anchor=(0.5, 0.02),
                   frameon=True, fontsize=8)
        
        if title:
            fig.suptitle(title, fontsize=10)
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        plt.show()

    @staticmethod
    def plot_error_growth(true_data, pred_data, title=None):
        """
        Computes and plots log-scale Euclidean error tracking.
        """
        steps = min(true_data.shape[1], pred_data.shape[1])
        errors = np.linalg.norm(true_data[:, :steps] - pred_data[:, :steps], axis=0)
        
        plt.figure(figsize=(7, 3.5))
        plt.semilogy(range(steps), errors, 'g-', linewidth=1.8)
        plt.xlabel('Discrete Step', fontsize=12)
        plt.ylabel('Error Norm (Log Scale)', fontsize=12)
        if title:
            plt.title(title, fontsize=11)
        plt.grid(True, which="both", alpha=0.3)
        plt.tick_params(labelsize=9)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_eigenvalues(eigenvalues, title=None):
        """
        Plots eigenvalues cleanly on the complex plane. 
        Legend identifies boundaries without blocking data points.
        """
        plt.figure(figsize=(5.5, 5.5))
        
        theta = np.linspace(0, 2 * np.pi, 200)
        plt.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.4, label='Unit Circle')
        plt.scatter(np.real(eigenvalues), np.imag(eigenvalues), c='crimson', 
                    marker='o', s=45, zorder=3, label=r'Eigenvalues $\mu$')
        
        plt.axhline(0, color='black', alpha=0.2, linestyle=':')
        plt.axvline(0, color='black', alpha=0.2, linestyle=':')
        plt.xlabel(r'$Re(\mu)$', fontsize=12)
        plt.ylabel(r'$Im(\mu)$', fontsize=12)
        if title:
            plt.title(title, fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right', frameon=True, fontsize=9)
        plt.axis('equal')
        plt.tick_params(labelsize=9)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_singular_values(S, r_trunc=None, title="Singular Value Spectrum"):
        """
        Publication-ready singular value spectrum plot.
        Optimized strictly for IEEE single-column width (3.5 inches).
        """
        fig, ax = plt.subplots(figsize=(7, 4.3))
        
        ax.semilogy(range(1, len(S) + 1), S, color='gray', linestyle='-', linewidth=0.8, zorder=1)
        
        if r_trunc is not None:
            # Retained Subspace: Solid dark blue markers
            ax.semilogy(range(1, r_trunc + 1), S[:r_trunc], 'o', color='#003366', 
                        markersize=4, label='Retained', zorder=3)
            # Discarded Subspace: Red crosses
            ax.semilogy(range(r_trunc + 1, len(S) + 1), S[r_trunc:], 'x', color='#cc0000', 
                        markersize=4, label='Discarded', zorder=3)
            
            # Truncation boundary
            ax.axvline(x=r_trunc + 0.5, color='black', linestyle='--', linewidth=0.8, alpha=0.8, zorder=2)
            
            # Subspace Shading (Light blue for physical, light red for noise)
            ax.axvspan(0, r_trunc + 0.5, color='#e6f2ff', alpha=0.4, lw=0, zorder=0)
            ax.axvspan(r_trunc + 0.5, len(S) + 1, color='#ffeeee', alpha=0.4, lw=0, zorder=0)
            
        ax.set_xlabel('Singular Value Index', fontsize=12)
        ax.set_ylabel('Magnitude (Log)', fontsize=12)
        if title:
            ax.set_title(title, fontsize=10)
            
        ax.grid(True, which="major", axis="y", color='gray', alpha=0.2, linestyle='-')
        ax.grid(True, which="major", axis="x", color='gray', alpha=0.05, linestyle='-')
        
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.set_xlim(0.5, len(S) + 0.5)
        
        if r_trunc is not None:
            ticks = [1, r_trunc] + list(range(5, len(S) + 1, 5))
            ax.set_xticks(sorted(list(set(ticks))))
        
        ax.legend(fontsize=12, loc='upper right', framealpha=0.0, edgecolor='none')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.show()

