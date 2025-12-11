# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 16:49:07 2025

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# System parameters (critical case T0S = 1)
# -----------------------------
mu1, mu2 = 0.45, 0.05
gamma1, gamma2 = 0.35, 0.15
beta1, beta2 = 0.006, 0.005
N = 100

# Markov switching rates
nu12 = 0.6   # 1 → 2
nu21 = 0.9   # 2 → 1

# Simulation settings
T = 300
dt = 0.01
steps = int(T / dt)

# Initial condition
I0 = 50
r0 = 1

# Storage
t_vals = np.linspace(0, T, steps)
I_vals = np.zeros(steps)
r_vals = np.zeros(steps)

I = I0
r = r0

I_vals[0] = I
r_vals[0] = r


# -----------------------------
# Markov chain transition
# -----------------------------
def switch_regime(state):
    if state == 1:
        if np.random.rand() < nu12 * dt:
            return 2
    else:
        if np.random.rand() < nu21 * dt:
            return 1
    return state


# -----------------------------
# Euler–Maruyama simulation
# -----------------------------
for k in range(1, steps):

    # Update Markov chain
    r = switch_regime(r)
    
    # Choose parameters for regime
    if r == 1:
        mu, gamma, beta = mu1, gamma1, beta1
    else:
        mu, gamma, beta = mu2, gamma2, beta2
    
    # Drift & diffusion
    drift = beta * (N - I) * I - (mu + gamma) * I
    diffusion = np.sqrt(max(beta * (N - I) * I, 0))

    # Brownian increment
    dW = np.sqrt(dt) * np.random.randn()

    # EM step
    I = I + drift * dt + diffusion * dW

    # Keep I inside [0, N]
    I = max(0, min(N, I))

    I_vals[k] = I
    r_vals[k] = r


# -----------------------------
# Plot I(t)
# -----------------------------
plt.figure(figsize=(10,5))
plt.plot(t_vals, I_vals, label="I(t) (Euler–Maruyama)")
plt.xlabel("Time")
plt.ylabel("Infected I(t)")
plt.title("SIS Model with Markov Switching — Critical Case (T0S = 1)")
plt.grid()
plt.legend()
plt.show()

# -----------------------------
# Plot regime r(t)
# -----------------------------
plt.figure(figsize=(10,3))
plt.plot(t_vals, r_vals, drawstyle='steps-post')
plt.yticks([1,2])
plt.xlabel("Time")
plt.title("Markov Chain Regime r(t)")
plt.grid()
plt.show()
