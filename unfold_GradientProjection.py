import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from matplotlib import pyplot as plt
# Config device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# ==============================
# 1. Generate channel data (Rayleigh fading)
# ==============================
def generate_channel(Nt, K, batch_size=100, d_min=10, d_max=100, correlation=0.2):
    """Generate realistic channel with path loss, shadowing and correlation"""
    # 1. Generate distances and large-scale effects
    d = torch.rand(batch_size, K, device=device) * (d_max - d_min) + d_min
    path_loss = 1.0 / (d ** 1.5)
    shadowing = 10 ** (torch.randn(batch_size, K, device=device) * 0.4 / 10)
    # 2. Generate small-scale fading with correlation
    R = torch.eye(Nt, device=device, dtype=torch.complex64) * (1 - correlation) + correlation
    L = torch.linalg.cholesky(R)
    H_real = torch.randn(batch_size, Nt, K, device=device)
    H_imag = torch.randn(batch_size, Nt, K, device=device)
    H = torch.complex(H_real, H_imag) * np.sqrt(0.5)
    H = torch.einsum('ij,bjk->bik', L, H)
    # 3. Combine all effects
    path_loss = path_loss.unsqueeze(1)  # [batch, 1, K]
    shadowing = shadowing.unsqueeze(1)  # [batch, 1, K]
    H = H * torch.sqrt(path_loss * shadowing)
    return H
# ==============================
# 2. Define GPNet (Unfolded Gradient Projection)
# ==============================
class GPNet(nn.Module):
    def __init__(self, Nt, K, num_layers=10, P=1.0):
        super().__init__()
        self.Nt = Nt
        self.K = K
        self.P = torch.tensor(P, device=device)
        self.num_layers = num_layers
        # Initialize parameters with power-dependent scaling
        self.eta = nn.Parameter(torch.ones(num_layers, device=device) * (0.1 / torch.sqrt(self.P)))
        self.U = nn.ParameterList([
            nn.Parameter(torch.randn(Nt, Nt, device=device) * (0.01 / torch.sqrt(self.P)))
            for _ in range(num_layers)
        ])
        self.b = nn.ParameterList([
            nn.Parameter(torch.randn(Nt, K, device=device) * (0.01 / torch.sqrt(self.P)))
            for _ in range(num_layers)
        ])
    def forward(self, H):
        batch_size = H.shape[0]
        W = torch.zeros(batch_size, self.Nt, self.K, dtype=torch.complex64, device=device)
        for t in range(self.num_layers):
            W = W.clone().requires_grad_(True)
            # Compute rate and gradient
            rate = sum_rate(H, W, target_power=self.P)
            grad_W = torch.autograd.grad(rate.sum(), W, create_graph=True)[0]
            # Compute the update
            grad_W_real = torch.view_as_real(grad_W)
            step_real = self.U[t] @ grad_W_real[..., 0] - self.U[t] @ grad_W_real[..., 1]
            step_imag = self.U[t] @ grad_W_real[..., 1] + self.U[t] @ grad_W_real[..., 0]
            step = torch.complex(step_real, step_imag)
            # Update W with power-scaled parameters
            W = W - self.eta[t] * step + torch.complex(self.b[t], torch.zeros_like(self.b[t]))
            # Only apply power constraint at the final layer
            if t == self.num_layers - 1:
                norm = torch.norm(W, dim=(1, 2), keepdim=True)
                scale = torch.sqrt(self.P) / (norm + 1e-6)
                W = W * scale
        return W
# Sum-rate calculation function
def sum_rate(H, W, noise_power=0.02, target_power=None):
    noise_power = torch.tensor(noise_power, device=device)
    batch_size, Nt, K = H.shape
    # Compute all user signals at once
    HW = torch.einsum('bnk,bnk->bk', H.conj(), W)
    desired_signal = torch.abs(HW) ** 2
    # Compute interference
    interference = torch.abs(torch.einsum('bnk,bnl->bkl', H.conj(), W)) ** 2
    interference = interference.sum(dim=2) - desired_signal
    sinr = desired_signal / (interference + noise_power)
    rates = torch.log2(1 + sinr)
    # Add power regularization term if target_power is provided
    if target_power is not None:
        current_power = torch.norm(W, dim=(1, 2)) ** 2
        power_penalty = torch.mean((current_power - target_power) ** 2) * 0.01
        return torch.sum(rates, dim=1) - power_penalty
    return torch.sum(rates, dim=1)
# ==============================
# 3. Traditional methods (ZF and WMMSE)
# ==============================
def ZF_beamforming(H, P=10):
    P = torch.tensor(P, device=device)
    H_H = H.conj().transpose(1, 2)
    W = H @ torch.linalg.inv(H_H @ H)
    norm_W = torch.norm(W, dim=(1, 2), keepdim=True)
    W = W / norm_W * torch.sqrt(P)
    return W
def WMMSE_beamforming(H, P=10, max_iter=50):
    P = torch.tensor(P, device=device)
    batch_size, Nt, K = H.shape
    W = torch.randn(batch_size, Nt, K, dtype=torch.complex64, device=device)
    for _ in range(max_iter):
        # Update weights
        V = torch.linalg.inv(H.conj().transpose(1, 2) @ H + torch.eye(K, device=device) * (K / P))
        W = H @ V
        # Normalize power
        norm_W = torch.norm(W, dim=(1, 2), keepdim=True)
        W = W / norm_W * torch.sqrt(P)
    return W
def GP_beamforming(H, P=10.0, max_iter=50, lr=0.01):
    """
    Gradient Projection (GP) for beamforming optimization.
    Args:
        H: Channel matrix [batch, Nt, K]
        P: Power constraint (should be float/tensor)
        max_iter: Number of iterations
        lr: Learning rate
    Returns:
        W: Beamforming matrix [batch, Nt, K]
    """
    batch_size, Nt, K = H.shape
    device = H.device
    # Convert P to tensor if it's not
    if not isinstance(P, torch.Tensor):
        P = torch.tensor(P, dtype=torch.float32, device=device)
    # Initialize W randomly
    W = torch.randn(batch_size, Nt, K, dtype=torch.complex64, device=device)
    W = W / torch.norm(W, dim=(1, 2), keepdim=True) * torch.sqrt(P)
    for _ in range(max_iter):
        W = W.clone().requires_grad_(True)
        # Compute sum-rate and gradient
        rate = sum_rate(H, W)
        grad_W = torch.autograd.grad(rate.sum(), W)[0]
        # Gradient ascent update
        W = W + lr * grad_W
        # Projection onto power constraint
        norm_W = torch.norm(W, dim=(1, 2), keepdim=True)
        W = W / norm_W * torch.sqrt(P)
    return W.detach()
# ==============================
# 4. Hybrid Training and evaluation
# ==============================
# System parameters
Nt = 64  # Number of antennas
K = 40  # Number of users
P = 1  # Transmit power
batch_size = 512
num_layers = 7
supervised_ratio = 0.5  # Ratio of supervised training samples
# Generate training and test data
H_train = generate_channel(Nt, K, batch_size)
H_test = generate_channel(Nt, K, 100)
# Generate WMMSE labels for supervised training (only do this once)
print("Generating WMMSE labels for supervised training...")
with torch.no_grad():
    W_wmmse = WMMSE_beamforming(H_train, P)
# Initialize model and optimizer
model = GPNet(Nt, K, num_layers, P).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Hybrid training function
def hybrid_train(model, H, W_labels, epochs=100, supervised_ratio=0.5):
    model.train()
    batch_size = H.shape[0]
    supervised_batch = int(batch_size * supervised_ratio)
    for epoch in range(epochs):
        # Shuffle the batch
        perm = torch.randperm(batch_size)
        H_shuffled = H[perm]
        W_shuffled = W_labels[perm]
        optimizer.zero_grad()
        # Forward pass for all samples
        W_pred = model(H_shuffled)
        # Supervised loss (MSE) for first part of batch
        supervised_loss = torch.mean(torch.abs(W_pred[:supervised_batch] - W_shuffled[:supervised_batch]) ** 2)
        # Unsupervised loss (negative sum-rate) for remaining part of batch
        unsupervised_loss = -sum_rate(H_shuffled[supervised_batch:], W_pred[supervised_batch:]).mean()
        # Combine losses with weighting
        total_loss = supervised_loss + unsupervised_loss
        # Backward pass
        total_loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            # Calculate current sum-rates for monitoring
            with torch.no_grad():
                rate_supervised = sum_rate(H_shuffled[:supervised_batch], W_pred[:supervised_batch]).mean()
                rate_unsupervised = sum_rate(H_shuffled[supervised_batch:], W_pred[supervised_batch:]).mean()
                rate_wmmse = sum_rate(H_shuffled, W_shuffled).mean()
            print(f"Epoch {epoch}:")
            print(f"  Supervised Loss: {supervised_loss.item():.4f} | "
                  f"Unsupervised Loss: {unsupervised_loss.item():.4f}")
            print(f"  Sup Rate: {rate_supervised.item():.2f} | "
                  f"Unsup Rate: {rate_unsupervised.item():.2f} | "
                  f"WMMSE Rate: {rate_wmmse.item():.2f}")
print("Starting hybrid training...")
hybrid_train(model, H_train, W_wmmse, epochs=150, supervised_ratio=supervised_ratio)
# Evaluation function
def evaluate(method, H):
    start_time = time.time()
    if method == "GPNet":
        W = model(H)
    elif method == "ZF":
        W = ZF_beamforming(H, P)
    elif method == "WMMSE":
        W = WMMSE_beamforming(H, P)
    elif method == 'GP_beamforming':
        W = GP_beamforming(H, P=torch.tensor(P, device=H.device))
    else:
        raise ValueError("Unknown method")
    time_elapsed = time.time() - start_time
    rate = sum_rate(H, W).mean().item()
    return rate, time_elapsed
# Compare methods
methods = ["GPNet", "ZF", "WMMSE","GP_beamforming"]
results = {method: evaluate(method, H_test) for method in methods}
# Print results
print("\nPerformance Comparison:")
for method, (rate, t) in results.items():
    print(f"{method:5s} | Sum-Rate: {rate:.2f} bps/Hz | Time: {t:.4f}s")
# Plot results
plt.figure(figsize=(10, 5))
plt.bar(results.keys(), [r[0] for r in results.values()], color=['blue', 'green', 'red'])
plt.title("Sum-Rate Comparison (Hybrid Training)")
plt.ylabel("bps/Hz")
plt.grid(True)
plt.show()
plt.figure(figsize=(10, 5))
plt.bar(results.keys(), [r[1] for r in results.values()], color=['blue', 'green', 'red'])
plt.title("Inference Time Comparison")
plt.ylabel("Seconds")
plt.grid(True)
plt.show()