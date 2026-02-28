import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

#Necessary functions for the t-SNE explanation demo, including data loading and matrix rotation.
from sample_tsne import tsne_sample_embedded_points
from utils import load_tabular_dataset, rotate_matrix

#We have to implement a handcrafted version of the BIR algorithm for local explanations.
def bir_local(Z_hd, Z_ld, alphas=[0.01, 0.1]):
    """
    Finds the best rotation angle that maximizes the R² of a Lasso regression explaining the t-SNE axis W1 with the HD variables.
    """
    best_r2 = -np.inf
    best_angle = 0
    best_weights = None

    #here we test all angles.
    for angle in range(0, 360, 5):
        R = rotate_matrix(angle)
        Z_rotated = Z_ld @ R.T
        #We fit a Lasso regression to explain the first t-SNE axis (W1) with the HD variables
        model = Lasso(alpha=alphas[0],max_iter=5000)
        model.fit(Z_hd, Z_rotated[:, 0])
        score = model.score(Z_hd, Z_rotated[:, 0]) #R² score
        if score > best_r2:
            best_r2 = score
            best_angle = angle
            best_weights = model.coef_
    return best_angle, best_weights, best_r2
if not os.path.exists('logs'): os.makedirs('logs')
X, labels, feature_names = load_tabular_dataset(dataset_name="country", standardize=True)

#Here we generate the SMOTE samples around the target point (which is Spain here) and we project them (only the synthetic points) in the t-SNE space.
target_idx = 121
target_name = labels[target_idx]
print(f"Explaining: {target_name} (index {target_idx})")
Y, Z_hd, Z_ld = tsne_sample_embedded_points(
    X, 
    selected_idx=target_idx,
    n_samples=100, 
    tsne_hyper_params={'perplexity': 10, 'random_state': 42},
    early_stop_hyper_params={'n_iter': 300},
    sampling_method="generate_samples_SMOTE",
    log_dir="logs",
    force_recompute=True
)

angle, weights, r2 = bir_local(Z_hd, Z_ld)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
ax1.scatter(Y[:, 0], Y[:, 1], c='lightgrey', alpha=0.5)
ax1.scatter(Y[target_idx, 0], Y[target_idx, 1], c='red', s=100, label=target_name)
ax1.scatter(Z_ld[:, 0], Z_ld[:, 1], c='blue', s=15, alpha=0.3, label='SMOTE')
ax1.set_title("t-SNE neighborhood")
ax1.legend()
top_idx = np.argsort(np.abs(weights))[-10:]
ax2.barh(np.array(feature_names)[top_idx], weights[top_idx], color='teal')
ax2.set_title(f"Interpretation (Rotation: {angle}°, R²={r2:.2f})")
plt.tight_layout()
plt.show()