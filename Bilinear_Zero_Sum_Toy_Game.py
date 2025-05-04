import numpy as np
import matplotlib.pyplot as plt

def simulate_bilinear_game(theta0=0.0, phi0=0.0, lr_model=0.1, lr_adversary=0.1, steps=100, alternate_update=False):
    """
    Simulate an adversarial learning dynamic in a bilinear zero-sum game: u(theta, phi) = theta * phi.
    The model ascends on u, the adversary ascends on -u (equivalently descends on u).
    Args:
        theta0, phi0: Initial parameters for model (theta) and adversary (phi).
        lr_model, lr_adversary: Learning rates for the model and adversary updates.
        steps: Number of iterations to simulate.
        alternate_update: If True, use alternating updates (model update then adversary update each step);
                          If False, perform simultaneous updates using gradients from the previous step.
    Returns:
        theta_history, phi_history: Arrays of parameter values over time (length steps+1).
    """
    theta, phi = theta0, phi0
    theta_history = [theta]
    phi_history = [phi]
    for t in range(steps):
        # Compute gradients of u = theta * phi
        dtheta = phi         # ∂u/∂θ
        dphi = theta         # ∂u/∂φ
        if alternate_update:
            # Update model then adversary (Stackelberg-type sequential update)
            theta += lr_model * dtheta
            # Re-compute gradient for adversary after model moves
            dphi = theta
            phi   -= lr_adversary * dphi
        else:
            # Simultaneous update (using gradients from current state)
            theta += lr_model * dtheta
            phi   -= lr_adversary * dphi
        theta_history.append(theta)
        phi_history.append(phi)
    return np.array(theta_history), np.array(phi_history)

# Example usage:
theta_hist, phi_hist = simulate_bilinear_game(theta0=1.0, phi0=0.0, lr_model=0.05, lr_adversary=0.05, steps=200, alternate_update=False)

# Plot theta and phi over iterations to observe dynamics
plt.figure(figsize=(6,4))
plt.plot(theta_hist, label='theta (model)')
plt.plot(phi_hist, label='phi (adversary)')
plt.xlabel('Iteration')
plt.ylabel('Parameter value')
plt.title('Bilinear Game Training Dynamics')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
