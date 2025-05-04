# toy2_sigmoid.py
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

w = 0.0    # model parameter
t = 0.0    # adversary threshold parameter
lr = 0.1
history_w, history_t = [], []

for it in range(200):
    # Compute outputs
    p = sigmoid(w)
    a = sigmoid(t)
    # Soft comparison: adversary "success" probability
    q = sigmoid(5*(p - a))  # steep sigmoid
    # Rewards (zero-sum)
    r_model = 1 - 2*q
    r_adv   = -r_model
    # Compute gradients (chain rule)
    dp_dw = p*(1-p)
    da_dt = a*(1-a)
    # ∂r_model/∂p = -2 * q * (1-q) * 5
    drdp = -2 * (q * (1 - q)) * 5
    # ∂r_model/∂a = -drdp  (since q = σ(5*(p-a)))
    drda = -drdp
    # Gradient of reward w.r.t parameters
    grad_w = drdp * dp_dw
    grad_t = drda * da_dt
    # Update (gradient ascent for each)
    w += lr * grad_w
    t += lr * (-grad_t)  # adversary maximizes -r_model
    history_w.append(sigmoid(w))
    history_t.append(sigmoid(t))

import matplotlib.pyplot as plt
plt.plot(history_w, label='Model output σ(w)')
plt.plot(history_t, label='Threshold σ(t)')
plt.legend()
plt.xlabel('Iteration'); plt.ylabel('Value')
plt.title('Toy2: Sigmoid-Threshold Oscillation')
plt.savefig('toy2_oscillation.png')
