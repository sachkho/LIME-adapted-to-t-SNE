# modified_tsne.py corrigé pour sklearn moderne
from time import time
import numpy as np
import sklearn
import sklearn.manifold
from sklearn.manifold import TSNE
from sklearn.manifold import _barnes_hut_tsne
from scipy import linalg
from scipy.spatial.distance import pdist, squareform

MACHINE_EPSILON = np.finfo(np.double).eps

def my_kl_divergence(params, P, degrees_of_freedom, n_samples, n_components,
                     skip_num_points=0, compute_error=True, **kwargs): # Ajout **kwargs
    X_embedded = params.reshape(n_samples, n_components)
    dist = pdist(X_embedded, "sqeuclidean")
    dist /= degrees_of_freedom
    dist += 1.0
    dist **= (degrees_of_freedom + 1.0) / -2.0
    Q = np.maximum(dist / (2.0 * np.sum(dist)), MACHINE_EPSILON)

    if compute_error:
        kl_divergence = 2.0 * np.dot(P, np.log(np.maximum(P, MACHINE_EPSILON) / Q))
    else:
        kl_divergence = np.nan

    grad = np.zeros((n_samples, n_components), dtype=params.dtype)
    PQd = squareform((P - Q) * dist)
    for i in range(skip_num_points, n_samples):
        grad[i] = np.dot(np.ravel(PQd[i], order="K"), X_embedded[i] - X_embedded)
    
    grad = grad.ravel()
    c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    grad *= c
    return kl_divergence, grad

def my_kl_divergence_bh(params, P, degrees_of_freedom, n_samples, n_components,
                        angle=0.5, skip_num_points=0, verbose=False, **kwargs): # Ajout **kwargs
    params = params.astype(np.float32, copy=False)
    X_embedded = params.reshape(n_samples, n_components)
    val_P = P.data.astype(np.float32, copy=False)
    neighbors = P.indices.astype(np.int64, copy=False)
    indptr = P.indptr.astype(np.int64, copy=False)
    grad = np.zeros(X_embedded.shape, dtype=np.float32)
    
    error = _barnes_hut_tsne.gradient(
        val_P, X_embedded, neighbors, indptr, grad, angle, n_components, verbose,
        dof=degrees_of_freedom)

    grad[:skip_num_points] = 0.0 # On fige les points (Section 4.2) 
    c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    grad = grad.ravel() * c
    return error, grad

def modifiedTSNE(**kwargs):
    # Patch des fonctions de calcul de gradient
    sklearn.manifold._t_sne._kl_divergence = my_kl_divergence
    sklearn.manifold._t_sne._kl_divergence_bh = my_kl_divergence_bh
    
    tsne = sklearn.manifold.TSNE(**kwargs)
    
    # --- PATCH POUR SKLEARN MODERNE ---
    # On injecte les attributs internes attendus par les versions récentes
    tsne._max_iter = kwargs.get('n_iter', 1000)
    tsne._EXPLORATION_MAX_ITER = 250
    return tsne