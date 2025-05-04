import numpy as np
import matplotlib.pyplot as plt

# Define the Agent class for the bilinear game
class Agent:
    def __init__(self, lr=0.05, init_mean=0.0, std=1.0):
        self.mu = init_mean
        self.std = std
        self.lr = lr
    def sample(self):
        return np.random.normal(self.mu, self.std)
    def update(self, action, reward):
        # REINFORCE gradient update for mean
        grad = (action - self.mu) * reward / (self.std**2)
        self.mu += self.lr * grad

# Experiment parameters
n_runs = 10   # Number of independent runs
n_iters = 100  # Iterations per run

# Storage for trajectories
model_arr = np.zeros((n_runs, n_iters))
adv_arr = np.zeros((n_runs, n_iters))

# Conduct multiple runs
for run in range(n_runs):
    model = Agent(lr=0.05, init_mean=1.0)
    adv   = Agent(lr=0.05, init_mean=-1.0)
    for it in range(n_iters):
        x = model.sample()
        y = adv.sample()
        # Payoffs
        r_model = x * y
        r_adv = -r_model
        # Update agents
        model.update(x, r_model)
        adv.update(y, r_adv)
        # Record means
        model_arr[run, it] = model.mu
        adv_arr[run, it] = adv.mu

# Compute average trajectories
model_mean = model_arr.mean(axis=0)
adv_mean = adv_arr.mean(axis=0)

# Plot average convergence
plt.figure(figsize=(8, 4))
plt.plot(model_mean, label='Model μ (avg over 10 runs)')
plt.plot(adv_mean,   label='Adversary μ (avg over 10 runs)')
plt.hlines(0, 0, n_iters, linestyles='dashed', colors='gray')
plt.xlabel('Iteration')
plt.ylabel('Parameter Mean')
plt.title('Toy1: Bilinear Game Average Convergence (10 runs)')
plt.legend()
plt.tight_layout()
plt.show()
