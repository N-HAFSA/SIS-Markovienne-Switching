import numpy as np
import matplotlib.pyplot as plt

# =============================
#     PARAMETERS
# =============================
alpha1 = -0.7
alpha2 =  0.2

beta1 = 0.001
beta2 = 0.004

N = 100

# Switching intensities
nu12 = 0.6
nu21 = 0.9

# Stationary distribution
pi1 = nu21 / (nu12 + nu21)
pi2 = nu12 / (nu12 + nu21)

print("pi1 =", pi1, "pi2 =", pi2)
print("alpha1*pi1 + alpha2*pi2 =", alpha1*pi1 + alpha2*pi2)

# Noise coefficient for SDE
sigma = 0.1

# Simulation settings
dt = 0.001
T = 60
times = np.arange(0, T, dt)
n = len(times)


# =============================
#   Markov chain r(t)
# =============================
def simulate_regime(seed=0):
    np.random.seed(seed)
    r = np.zeros(n, dtype=int)
    r[0] = 1  # start in regime 1

    for k in range(1, n):
        if r[k-1] == 1:
            r[k] = 2 if np.random.rand() < nu12 * dt else 1
        else:
            r[k] = 1 if np.random.rand() < nu21 * dt else 2

    return r


# =============================
#   Formula (2.5) – deterministic solution
# =============================
def solve_formula_25(I0, r):
    I = np.zeros(n)
    I[0] = I0
    for k in range(1, n):
        if r[k] == 1:
            alpha, beta = alpha1, beta1
        else:
            alpha, beta = alpha2, beta2

        # Logistic-type ODE: dI = alpha*I - beta*I^2
        I[k] = I[k-1] + dt*(alpha*I[k-1] - beta*I[k-1]**2)
        I[k] = max(0, min(I[k], N))
    return I


# =============================
#   Euler–Maruyama Scheme
# =============================
def solve_EM(I0, r):
    I = np.zeros(n)
    I[0] = I0
    for k in range(1, n):
        if r[k] == 1:
            alpha, beta = alpha1, beta1
        else:
            alpha, beta = alpha2, beta2

        dW = np.sqrt(dt) * np.random.randn()
        I[k] = (
            I[k-1]
            + dt*(alpha*I[k-1] - beta*I[k-1]**2)
            + sigma * I[k-1] * dW
        )
        I[k] = max(0, min(I[k], N))
    return I


# =============================
#     RUN SIMULATIONS
# =============================
r = simulate_regime(seed=1)
I_formula = solve_formula_25(10, r)
I_em = solve_EM(10, r)


# =============================
#     PLOT I(t)
# =============================
plt.figure(figsize=(10, 5))
plt.plot(times, I_formula, color='black', linewidth=2, label="Formula (2.5)")
plt.plot(times, I_em, color='red', alpha=0.7, label="Euler–Maruyama")
plt.xlabel("Time")
plt.ylabel("I(t)")
plt.title("Extinction of SIS model with Markovian switching")
plt.grid(True)
plt.legend()
plt.show()


# =============================
#     PLOT r(t)
# =============================
plt.figure(figsize=(10, 2.5))
plt.step(times, r, where='post')
plt.xlabel("Time")
plt.ylabel("r(t)")
plt.yticks([1, 2])
plt.title("Regime process r(t)")
plt.grid(True)
plt.show()
