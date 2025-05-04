import numpy as np
import matplotlib.pyplot as plt
def simulate_threshold_game(theta0=0.5, phi0=0.7, lr_model=0.05, lr_adversary=0.05, steps=200,
                             lambd=1.0, p_benign=0.5, benign_mean=0.3, benign_std=0.1,
                             mal_std=0.1, sigmoid_k=50, batch_size=1000, seed=None):
    """
    Simulate adversarial training in a threshold-classification game between a model and an adversary.
    Model parameter theta: threshold for answering (higher theta = more lenient).
    Adversary parameter phi: controls malicious prompt 'score' distribution (lower phi = more benign-looking prompts).
    
    Args:
        theta0, phi0: Initial threshold and adversary parameter.
        lr_model, lr_adversary: Learning rates for model and adversary updates.
        steps: Number of training iterations.
        lambd: Penalty weight lambda for malicious answers in model utility.
        p_benign: Proportion of benign prompts (the rest 1-p_benign are malicious).
        benign_mean, benign_std: Mean and std dev of benign prompt score distribution.
        mal_std: Std dev of malicious prompt score distribution (its mean is controlled by phi).
        sigmoid_k: Sigmoid steepness (higher = closer to hard threshold).
        batch_size: Number of prompt samples (per type) to use for gradient estimation each iteration.
        seed: Random seed for reproducibility (optional).
    Returns:
        theta_history, phi_history: Arrays of the model threshold and adversary parameter over time.
    """
    if seed is not None:
        np.random.seed(seed)
    theta, phi = theta0, phi0
    theta_history = [theta]
    phi_history = [phi]
    q = p_benign
    for t in range(steps):
        # Sample benign and malicious prompt scores
        benign_scores = np.random.normal(benign_mean, benign_std, batch_size)
        mal_scores    = np.random.normal(phi, mal_std, batch_size)
        benign_scores = np.clip(benign_scores, 0.0, 1.0)  # scores bounded in [0,1]
        mal_scores    = np.clip(mal_scores, 0.0, 1.0)
        # Model's probability of answering (sigmoid threshold policy)
        p_answer_ben = 1.0 / (1.0 + np.exp(-sigmoid_k * (theta - benign_scores)))
        p_answer_mal = 1.0 / (1.0 + np.exp(-sigmoid_k * (theta - mal_scores)))
        # Estimate fractions (rates) of prompts answered
        benign_answer_rate = p_answer_ben.mean()
        malicious_answer_rate = p_answer_mal.mean()
        # Compute gradients:
        # dU/dθ = q * E[d/dθ σ(theta - x_ben)] - λ*(1-q) * E[d/dθ σ(theta - x_mal)]
        grad_theta = q * np.mean(sigmoid_k * p_answer_ben * (1 - p_answer_ben)) \
                     - lambd * (1 - q) * np.mean(sigmoid_k * p_answer_mal * (1 - p_answer_mal))
        # d(AdversaryObj)/dφ = d(malicious_answer_rate)/dφ.
        # Since mal_scores = φ + noise, ∂/∂φ σ(theta - x) = -∂/∂x σ = -sigmoid_k * p*(1-p).
        grad_phi = np.mean(-sigmoid_k * p_answer_mal * (1 - p_answer_mal))
        # Gradient ascent for model, ascent for adversary's objective (so descent on phi for model's utility).
        theta += lr_model * grad_theta
        phi   += lr_adversary * grad_phi
        # Clamp theta and phi to [0,1] (valid score range)
        theta = max(0.0, min(1.0, theta))
        phi   = max(0.0, min(1.0, phi))
        theta_history.append(theta)
        phi_history.append(phi)
    return np.array(theta_history), np.array(phi_history)

# Example usage:
theta_hist, phi_hist = simulate_threshold_game(theta0=0.5, phi0=0.7, lr_model=0.05, lr_adversary=0.05,
                                              steps=300, lambd=1.0, p_benign=0.5, seed=42)
# Plot the parameter trajectories
plt.figure(figsize=(6,4))
plt.plot(theta_hist, label='theta (model threshold)')
plt.plot(phi_hist, label='phi (malicious prompt level)')
plt.xlabel('Iteration')
plt.ylabel('Parameter value')
plt.title('Threshold Policy Training Dynamics')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
