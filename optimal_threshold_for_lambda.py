import math
import numpy as np
import matplotlib.pyplot as plt
def optimal_threshold_for_lambda(lam, p_benign=0.5, benign_mean=0.3, benign_std=0.1,
                                 malicious_mean=0.6, malicious_std=0.1):
    """Find the optimal threshold that maximizes U = p_benign*benign_rate - lam*(1-p_benign)*malicious_rate."""
    best_utility = -1e9
    best_th = 0.0
    best_rates = (0.0, 0.0)
    # We discretize threshold from 0 to 1
    for th in np.linspace(0, 1, 501):
        # Compute benign and malicious answer rates for threshold th (using CDF of normal dist)
        benign_rate = 0.5 * (1 + math.erf((th - benign_mean) / (benign_std * math.sqrt(2))))
        mal_rate    = 0.5 * (1 + math.erf((th - malicious_mean) / (malicious_std * math.sqrt(2))))
        # Utility = p_benign*benign_rate - lam*(1-p_benign)*mal_rate
        utility = p_benign * benign_rate - lam * (1 - p_benign) * mal_rate
        if utility > best_utility:
            best_utility = utility
            best_th = th
            best_rates = (benign_rate, mal_rate)
    return best_th, best_rates

# Sweep lambda values and collect trade-off points
lambda_values = np.linspace(0.1, 10, 50)  # (start at 0.1 to avoid the trivial lambda=0 case)
benign_rates = []
malicious_rates = []
for lam in lambda_values:
    _, (b_rate, m_rate) = optimal_threshold_for_lambda(lam)
    benign_rates.append(b_rate)
    malicious_rates.append(m_rate)

# Plot the trade-off curve (malicious vs benign answer rates)
plt.figure(figsize=(6,5))
plt.plot(np.array(malicious_rates)*100, np.array(benign_rates)*100, marker='o', color='orange')
plt.xlabel('Malicious Prompt Answer Rate (%)')
plt.ylabel('Benign Prompt Answer Rate (%)')
plt.title('Utility-Safety Trade-off as λ varies')
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.show()
