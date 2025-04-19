# toy_games_simulation.py

import math
import numpy as np
import matplotlib.pyplot as plt

def train_bilinear_game(lr_theta=0.1, lr_phi=0.1, iterations=100, theta0=0.5, phi0=0.5):
    """
    Train the censorship game in a simple bilinear zero-sum setting.
    Utility: U_A = theta * phi, U_M = - theta * phi.
    Returns history of (theta, phi).
    """
    theta = theta0
    phi = phi0
    history = []
    for t in range(iterations):
        # Gradients for bilinear game:
        # d(U_M)/dtheta = -phi  (model wants to maximize U_M, so gradient ascent step uses +dU_M)
        # d(U_A)/dphi = theta  (adversary wants to maximize U_A)
        # Model update (ascent on U_M):
        theta_grad = -phi
        theta += lr_theta * theta_grad
        # Project theta to [0,1]
        theta = max(0.0, min(1.0, theta))
        # Adversary update (ascent on U_A):
        phi_grad = theta  # derivative of theta*phi w.rt phi
        phi += lr_phi * phi_grad
        phi = max(0.0, min(1.0, phi))
        history.append((theta, phi))
    return np.array(history)

def train_threshold_game(p=0.1, lam=5.0, lr_theta=0.1, lr_phi=0.1, iterations=100, theta0=0.5, phi0=0.5):
    """
    Train the censorship game in the sigmoid threshold setting.
    p: fraction of prompts that are malicious.
    lam: lambda (penalty weight for violations in model utility).
    Model's policy param: theta (threshold).
    Adversary's param: phi (malicious prompt "explicitness").
    Returns history of (theta, phi, model_util, adv_util, P_answer_mal, P_answer_ben).
    """
    theta = theta0
    phi = phi0
    # Benign prompt distribution: we model x ~ Beta(2,5) (skewed to low values)
    benign_alpha, benign_beta = 2.0, 5.0
    history = []
    for t in range(iterations):
        # The adversary chooses a malicious prompt with "maliciousness" = phi (deterministic).
        x_mal = phi
        # Model's probability to answer given x:
        # Using sigmoid with slope k:
        k = 10.0
        p_answer_mal = 1.0 / (1.0 + math.exp(-k * (theta - x_mal)))
        # For benign, approximate expectation via sampling:
        x_benign_samples = np.random.beta(benign_alpha, benign_beta, size=50)
        p_answer_benign_samples = 1.0 / (1.0 + np.exp(-k * (theta - x_benign_samples)))
        p_answer_ben = float(np.mean(p_answer_benign_samples))
        # Compute utilities:
        # R_task = 1 for benign answered, 0 if refused or if prompt malicious.
        U_model = (1-p) * (1.0 * p_answer_ben) + p * (0.0 - lam * p_answer_mal)
        U_adv = p_answer_mal * phi  # adversary reward = x * (if answered)
        # Record metrics:
        history.append((theta, phi, U_model, U_adv, p_answer_mal, p_answer_ben))
        # Gradients:
        d_p_answer_ben = np.mean(10.0 * p_answer_benign_samples * (1 - p_answer_benign_samples))
        d_p_answer_mal = 10.0 * p_answer_mal * (1 - p_answer_mal)
        dU_model_dtheta = (1-p) * 1.0 * d_p_answer_ben + p * (-lam) * d_p_answer_mal
        # dU_adv/dphi = p_answer_mal + phi * d(p_answer_mal)/dphi, and d(p_answer_mal)/dphi = -d_p_answer_mal (since derivative inside is -k).
        d_p_mal_d_phi = -d_p_answer_mal
        dU_adv_dphi = p_answer_mal + phi * d_p_mal_d_phi
        # Policy gradient ascent (model):
        theta += lr_theta * dU_model_dtheta
        # Adversary gradient ascent:
        phi += lr_phi * dU_adv_dphi
        # Clamp theta and phi to [0,1]:
        theta = max(0.0, min(1.0, theta))
        phi = max(0.0, min(1.0, phi))
    return np.array(history)

if __name__ == "__main__":
    # Example usage and generating plots
    
    # 1. Train bilinear game
    hist = train_bilinear_game(lr_theta=0.05, lr_phi=0.05, iterations=20)
    print("Bilinear game equilibrium (approx): theta =", hist[-1,0], "phi =", hist[-1,1])
    
    # 2. Train threshold game
    hist2 = train_threshold_game(p=0.1, lam=5.0, lr_theta=0.05, lr_phi=0.05, iterations=100)
    print("Threshold game final: theta =", hist2[-1,0], "phi =", hist2[-1,1])
    
    # Plot convergence of threshold game parameters
    plt.figure()
    plt.plot(hist2[:,0], label="Model threshold (theta)")
    plt.plot(hist2[:,1], label="Adversary prompt parameter (phi)")
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.title("Convergence in Threshold Game (p=0.1, lambda=5)")
    plt.legend()
    plt.savefig("threshold_game_convergence.png")
    
    # Plot trade-off: varying lambda outcomes
    lam_values = [0.001, 0.1, 0.5, 1, 2, 5, 10, 50, 100]
    ben_rates = []
    mal_rates = []
    for lam in lam_values:
        h = train_threshold_game(p=0.1, lam=lam, lr_theta=0.1, lr_phi=0.1, iterations=200)
        p_ans_ben = h[-1,5]
        p_ans_mal = h[-1,4]
        ben_rates.append(p_ans_ben*100)
        mal_rates.append(p_ans_mal*100)
    plt.figure()
    plt.plot(ben_rates, mal_rates, marker='o')
    for i, lam in enumerate(lam_values):
        plt.annotate(f"lambda={lam}", (ben_rates[i]+1, mal_rates[i]+1))
    plt.xlabel("Benign Answer Rate (%)")
    plt.ylabel("Malicious Prompt Success Rate (%)")
    plt.title("Utility-Safety Trade-off Curve")
    plt.savefig("utility_safety_tradeoff.png")
