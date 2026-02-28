# t-SNE Local Explanation (BIR Algorithm)

This repository provides a compact and functional implementation of the local explanation method for t-SNE embeddings, as proposed by **Bibal et al. (2020)**. 

The project demonstrates how to make t-SNE—a non-parametric and non-linear dimensionality reduction technique—locally interpretable through **synthetic sampling** and **optimal rotation (BIR)**.

## Overview

The core objective is to explain the "local logic" behind a specific point's position in a t-SNE plot (e.g., why Spain is located in a specific cluster). Since t-SNE axes are arbitrary and rotation-invariant, we use the **Best Interpretable Rotation (BIR)** algorithm to find an orientation that aligns the local neighborhood with original high-dimensional features.

### Key Technical Features
* **Local SMOTE Sampling**: Generates synthetic points on the high-dimensional data manifold to probe the local structure.
* **Out-of-sample Projection**: A custom t-SNE gradient implementation (`skip_num_points`) allows projecting new samples without displacing original data points.
* **BIR Algorithm**: A systematic search for the rotation angle that maximizes the $R^2$ of a local Lasso regression, revealing variable importance.

## Installation & Usage

### Prerequisites
* Python 3.10+
* `scikit-learn`, `numpy`, `pandas`, `matplotlib`, `joblib`

### Running the Demo
The main script `report_code.py` performs the full pipeline: data loading, t-SNE embedding, local SMOTE generation, and BIR rotation.

```bash
python report_code.py
```

## 📊 Methodology

This implementation illustrates **Section 4.3 (BIR)** of the paper:
1.  **Selection**: Pick a target instance $x$ (e.g., Spain).
2.  **Sampling**: Generate $m$ synthetic samples $z_j$ around $x$ using a modified SMOTE approach.
3.  **Projection**: Project $z_j$ into the 2D space using a modified t-SNE where the gradient of existing points is forced to zero.
4.  **Rotation**: Iterate through angles $	heta \in [0, 360^\circ]$ to find the orientation where the axes are best explained by a linear combination of original features.

## Project Structure

* `report_code.py`: Main execution script (under 100 lines).
* `modified_tsne.py`: Modified Scikit-Learn t-SNE for fixed-point gradients.
* `sample_tsne.py`: Logic for out-of-sample projection.
* `sampling.py`: Neighborhood generation (SMOTE).
* `utils.py`: Data loading and mathematical rotation utilities.
* `dataset/`: Contains the `country.dat` binary data.

## Citation

If you use this work, please cite the original paper:

**APA:**
> Bibal, A., Lognoul, M., De Bruyne, C., de Terwangne, C., & Frénay, B. (2020). Legal requirements on explainability in AI. *Proceedings of the 28th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN)*, 547-552.

**BibTeX:**
```bibtex
@inproceedings{bibal2020legal,
  title={Legal requirements on explainability in AI},
  author={Bibal, Adrien and Lognoul, Michael and De Bruyne, Alexandre and de Terwangne, C{'e}cile and Fr{'e}nay, Beno{\^\i}t},
  booktitle={28th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN)},
  pages={547--552},
  year={2020}
}
```
