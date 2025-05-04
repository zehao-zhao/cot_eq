# toy2_tradeoff.py
import numpy as np

def sigmoid(x): return 1/(1+np.exp(-x))

# Simulate varying lambda (censorship weight)
lambdas = np.linspace(0, 5, 21)
tradeoff = []
for lam in lambdas:
    # Use equilibrium strategy (e.g., both converge to 0.5 output at lambda ~0)
    # For simplicity, assume model output ~1/(1+exp(-theta)), we sweep reward:
    # utility = p, safety = (1 - q). For a grid of p:
    best_util = []
    for w in np.linspace(-4,4,201):
        p = sigmoid(w)
        # cost if violation q = sigmoid(5*(p-0.5))
        q = sigmoid(5*(p-0.5))
        util = p - lam*q
        best_util.append(util)
    best_val = max(best_util)
    tradeoff.append(best_val)

import matplotlib.pyplot as plt
plt.plot(lambdas, tradeoff)
plt.xlabel('Censorship weight λ'); plt.ylabel('Max utility minus λ * violation')
plt.title('Toy2: Utility-Safety Tradeoff')
plt.savefig('tradeoff.png')
