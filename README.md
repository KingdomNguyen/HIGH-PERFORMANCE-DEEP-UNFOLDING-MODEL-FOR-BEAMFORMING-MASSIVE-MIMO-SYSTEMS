# HIGH PERFORMANCE DEEP UNFOLDING MODEL FOR BEAMFORMING IN MASSIVE MIMO SYSTEMS

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange)](https://pytorch.org/)

<img src="docs/architecture.png" alt="GPNet Architecture" width="600"/>

*Deep Unfolded Gradient Projection Network for  Beamforming Optimization*

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
- Weighted minimum mean-square error (WMMSE): High computational complexity because of terative optimization
- Gradient Projection (GP): Better theoretical guarantees but computationally expensive
## Methodology
### Problem formulation
Maximize sum-rate under power constraint: 


