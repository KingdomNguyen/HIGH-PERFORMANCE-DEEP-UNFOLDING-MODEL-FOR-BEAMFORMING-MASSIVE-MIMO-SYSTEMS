# HIGH PERFORMANCE DEEP UNFOLDING MODEL FOR BEAMFORMING IN MASSIVE MIMO SYSTEMS

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange)](https://pytorch.org/)



*Deep Unfolded Gradient Projection Network for fast Beamforming Optimization*

## Table of Contents
- [Problem Statement](#problem-statement)
- [Methodology](#methodology)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [References](#references)
- [License](#license)
## Problem Statement
This project addresses the critical challenge of beamforming optimization in massive MIMO (Multiple-Input Multiple-Output) systems to maximize sum-rate capacity while satisfying power constraints. The key difficulties include:
- High computational complexity of traditional optimization methods
- Real-time processing requirements for practical deployment
- Handling of complex channel conditions (path loss, shadowing, correlation)
  
There are some traditional methods:
- Zero-Forcing (ZF) Beamforming: Fast but performance drops when number of users is large
- Weighted minimum mean-square error (WMMSE): High computational complexity because of iterative optimization
- Gradient Projection (GP): Better theoretical guarantees but computationally expensive
## Methodology
### Problem formulation
- Objective: Optimize the beamforming matrix (W) to maximize the sum-rate in a MU-MIMO system under a transmit power constraint (P).
- Input: Channel matrix (H) of size [batch_size, Nt, K] (Nt: number of antennas, K: number of users).
- Output: Beamforming matrix (W) of size [batch_size, Nt, K] such that ||W||^2 ≤ P.
- Objective function: Sum-rate is computed using Shannon capacity based on each user’s SINR (see sum_rate() function in the code).
### Gradient Projection (GP) Algorithm
- Idea: Update W along the gradient of the sum-rate, then project it onto the power constraint.
- Steps (implemented in GP_beamforming()):
  1. Initialize W randomly.
  2. Compute the gradient of the sum-rate w.r.t. W using torch.autograd.grad.
  3. Update W: W = W + lr * grad_W.
  4. Project W to satisfy the power constraint: W = W / norm(W) * sqrt(P).
### Unfolded GP
- What is Unfolding?: Converting an iterative algorithm (like GP) into a fixed-depth neural network (NN), where each layer corresponds to one iteration. Parameters (e.g., learning rate, transformation matrices) are learned from data.
- Applied to GPNet (see GPNet class): 
  
  Each layer is a modified GP step:
  + Instead of a fixed learning rate, it uses a learnable parameter eta[t].
  + Adds a transformation matrix U[t] and bias b[t] to enhance flexibility.
  + The power constraint is enforced only at the final layer.

  Benefits: Unfolding allows GPNet to converge faster than traditional GP with fewer layers (due to learned optimization dynamics).
### Channel Model (Rayleigh Fading)
- Model (implemented in generate_channel()):
  + Path loss: Decays with distance (1/d^1.5).
  + Shadowing: Log-normal random variations.
  + Small-scale fading: Rayleigh-distributed with antenna correlation (modeled via Cholesky decomposition of correlation matrix R).
###  Hybrid Training
- Combines Supervised + Unsupervised Learning:
  + Supervised: Uses WMMSE as labels, with loss as MSE between predicted W and WMMSE (see hybrid_train()).
  + Unsupervised: Directly maximizes sum-rate (negative sum-rate as loss).
- Purpose: Leverages the strengths of both methods (WMMSE for good initialization, unsupervised for surpassing WMMSE performance).
## Installation
### Prerequisites
- Python 3.8+ (Tested with 3.10)
- CUDA 11.7+ (if using GPU)
- PyTorch
## Usage
### 1. Basic Training & Evaluation
```python
from model import GPNet
from utils import generate_channel, evaluate

# System parameters
Nt, K, P = 32, 10, 1.0  # Antennas, Users, Power

# Initialize model
model = GPNet(Nt=Nt, K=K, num_layers=10, P=P).cuda()

# Generate synthetic channels
H_train = generate_channel(Nt, K, batch_size=512)
H_test = generate_channel(Nt, K, batch_size=100)

# Train (hybrid supervised-unsupervised)
hybrid_train(model, H_train, epochs=150)

# Evaluate against benchmarks
results = evaluate("GPNet", H_test)
print(f"GPNet Sum-Rate: {results[0]:.2f} bps/Hz")
```
## Results
Some results in different cases:
- Nt = 32, K = 10, P = 1:
  ```
  Performance Comparison:
  GPNet | Sum-Rate: 7.34 bps/Hz | Time: 0.0367s
  ZF    | Sum-Rate: 2.87 bps/Hz | Time: 0.0106s
  WMMSE | Sum-Rate: 6.10 bps/Hz | Time: 0.0427s
  GP_beamforming | Sum-Rate: 7.77 bps/Hz | Time: 0.1017s
  ```
- Nt = 32, K = 20, P = 1: 
  ```
  Performance Comparison:
  GPNet | Sum-Rate: 10.96 bps/Hz | Time: 0.0501s
  ZF    | Sum-Rate: 0.28 bps/Hz | Time: 0.0000s
  WMMSE | Sum-Rate: 7.73 bps/Hz | Time: 0.1267s
  GP_beamforming | Sum-Rate: 11.56 bps/Hz | Time: 0.1603s
  ```
- Nt = 64, K=50, P = 1:
  ```
  Performance Comparison:
  GPNet | Sum-Rate: 20.28 bps/Hz | Time: 0.1189s
  ZF    | Sum-Rate: 1.86 bps/Hz | Time: 0.0051s
  WMMSE | Sum-Rate: 13.04 bps/Hz | Time: 0.2315s
  GP_beamforming | Sum-Rate: 19.90 bps/Hz | Time: 0.3670s
  ```
## References
[1] N. Samuel, T. Diskin, and A. Wiesel, “Learning to detect,” IEEE Trans.
Signal Process., vol. 67, no. 10, pp. 2554–2564, May 2019

[2] Q. Hu, Y. Cai, Q. Shi, K. Xu, G. Yu, and Z. Ding, “Iterative
algorithm induced deep-unfolding neural networks: Precoding design
for multiuser MIMO systems,” IEEE Trans. Wireless Commun., vol. 20,
no. 2, pp. 1394–1410, Feb. 2021.

[3] M. Zhu, T.-H. Chang, and M. Hong, “Learning to beamform in heterogeneous massive MIMO networks,” IEEE Trans. Wireless Commun.,
vol. 22, no. 7, pp. 4901–4915, Jul. 2023

