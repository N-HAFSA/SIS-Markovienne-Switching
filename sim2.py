# -*- coding: utf-8 -*-
"""
Created on Sat Dec  6 13:20:40 2025

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
mu = [0.45, 0.05]
gamma = [0.35, 0.15]
beta = [0.004, 0.012]
N = 100

# Markov switching rates
nu12 = 0.6   # from state 1 -> 2
nu21 = 0.9   # from state 2 -> 1

# Simulation parameters
T = 60
dt = 0.01
n = int(T / dt)

# Initial conditions
I = np.zeros(n)
I[0] = 50  # initial infected

r = np.zeros(n, dtype=int)
r[0] = 0   # starting environment (0 = state 1)

# Simulation loop
for t in range(1, n):

    # --- Markov switching ---
    if r[t-1] == 0:
        # state 1 -> 2
        if np.random.rand() < nu12 * dt:
            r[t] = 1
        else:
            r[t] = 0
    else:
        # state 2 -> 1
        if np.random.rand() < nu21 * dt:
            r[t] = 0
        else:
            r[t] = 1

    # SIS parameters for current state
    mu_r = mu[r[t]]
    gamma_r = gamma[r[t]]
    beta_r = beta[r[t]]

    # Deterministic SIS model (Euler)
    dI = I[t-1] * (beta_r * (N - I[t-1]) - mu_r - gamma_r)
    I[t] = I[t-1] + dI * dt

    # Bound conditions
    if I[t] < 0:
        I[t] = 0

# Time vector
time = np.linspace(0, T, n)

# === Plot 1: I(t) ===
plt.figure(figsize=(15, 10))

plt.subplot(2, 1, 1)
plt.plot(time, I, linewidth=1.2)
plt.title(" Cas a", fontsize=16)
plt.ylabel("I(t)", fontsize=14)
plt.grid(True)

# === Plot 2: r(t) ===
plt.subplot(2, 1, 2)
plt.plot(time, r, drawstyle="steps-post")
plt.title("r(t)", fontsize=16)
plt.xlabel("Time", fontsize=14)
plt.ylabel("r(t)", fontsize=14)
plt.yticks([0, 1], ["etat 1", "etat 2"])
plt.grid(True)

plt.tight_layout()
plt.show()
