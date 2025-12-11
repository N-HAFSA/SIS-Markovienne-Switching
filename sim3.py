# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 23:17:09 2025

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================
#   Parameters of the system
# =============================
mu1, mu2 = 0.45, 0.05
gamma1, gamma2 = 0.35, 0.15
beta1, beta2 = 0.004, 0.012
N = 100
nu12, nu21 = 0.6, 0.9

alpha1, alpha2 = -0.4, 1.0   # given
upper_bound = alpha2 / beta2  # = 83.33...

# =============================
#   Simulation settings
# =============================
dt = 0.01
T = 500
times = np.arange(0, T+dt, dt)
n_steps = len(times)


# =============================================================
#   Function: simulate SIS model with two-regime Markov chain
# =============================================================
def simulate_trajectory(I0, r0, seed=None):
    if seed is not None:
        np.random.seed(seed)

    I = np.zeros(n_steps)
    r = np.zeros(n_steps, dtype=int)

    I[0] = I0
    r[0] = r0

    for k in range(1, n_steps):

        # Markov switching for r(t)
        if r[k-1] == 1:
            r[k] = 2 if np.random.rand() < nu12 * dt else 1
        else:
            r[k] = 1 if np.random.rand() < nu21 * dt else 2

        # Choose parameters
        if r[k] == 1:
            alpha = alpha1
            beta = beta1
        else:
            alpha = alpha2
            beta = beta2

        # Euler step of logistic-type ODE
        I_new = I[k-1] + dt * (alpha * I[k-1] - beta * I[k-1]**2)

        # Keep inside domain
        I_new = max(0, min(I_new, N * 0.99999))
        I[k] = I_new

    return times, I, r


# =======================================
#   Run one trajectory for demonstration
# =======================================
times, I_traj, r_traj = simulate_trajectory(I0=10, r0=1, seed=42)


# =============================
#     Plot results
# =============================
plt.figure(figsize=(10, 5))
plt.plot(times, I_traj, label="I(t)")
plt.axhline(upper_bound, linestyle="--", label=f"alpha2/beta2 = {upper_bound:.2f}")
plt.xlabel("Time")
plt.ylabel("Infected I(t)")
plt.title("SIS Model with Markovian Switching")
plt.legend()
plt.grid(True)
plt.show()


# =============================
#     Plot regime r(t)
# =============================
plt.figure(figsize=(10, 2.5))
plt.plot(times, r_traj)
plt.xlabel("Time")
plt.ylabel("Regime")
plt.title("Regime r(t)")
plt.yticks([1, 2])
plt.grid(True)
plt.show()
