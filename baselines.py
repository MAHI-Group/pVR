"""
Baseline methods for comparison with pVR.
Implements: FFP (Jensen-Shannon), NVM (Natural Vector Method), MinHash (Mash-like).

Written by: Tirtharaj Dash (assistance from Claude Opus 4.x), Jan 2026; bug-fix: Apr 2026
"""

import numpy as np
from collections import Counter
from itertools import product as iter_product
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
import hashlib


def extract_kmers(seq, k):
    if len(seq) < k:
        return []
    return [seq[i:i + k] for i in range(len(seq) - k + 1)]


# ============================================================
# FFP: Feature Frequency Profile (Sims et al., PNAS 2009)
# ============================================================

def ffp_frequency_vector(seq, k, alphabet="ACGT"):
    """Normalized k-mer frequency vector."""
    kmers = extract_kmers(seq, k)
    counts = Counter(kmers)
    all_kmers = ["".join(x) for x in iter_product(alphabet, repeat=k)]
    total = max(len(kmers), 1)
    return np.array([counts.get(km, 0) / total for km in all_kmers])


def ffp_jensen_shannon_distance(sequences, k=3, alphabet="ACGT"):
    """
    FFP with Jensen-Shannon divergence.
    Sims et al., PNAS 2009.
    """
    N = len(sequences)
    freq_vectors = np.array([ffp_frequency_vector(s, k, alphabet) for s in sequences])
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            p = freq_vectors[i]
            q = freq_vectors[j]
            m = 0.5 * (p + q)
            kl_pm = np.sum(np.where(p > 0, p * np.log2(p / np.where(m > 0, m, 1e-12)), 0))
            kl_qm = np.sum(np.where(q > 0, q * np.log2(q / np.where(m > 0, m, 1e-12)), 0))
            js = 0.5 * (kl_pm + kl_qm)
            D[i, j] = np.sqrt(max(js, 0))
            D[j, i] = D[i, j]
    return D


def ffp_euclidean_distance(sequences, k=3, alphabet="ACGT"):
    """FFP with Euclidean distance (simpler variant)."""
    freq_vectors = np.array([ffp_frequency_vector(s, k, alphabet) for s in sequences])
    return squareform(pdist(freq_vectors, metric="euclidean"))


# ============================================================
# NVM: Natural Vector Method (Deng et al., PLoS ONE 2011)
# ============================================================
def nvm_features(seq, k=1):
    """
    Natural Vector Method (Deng et al., PLoS ONE, 2011).
    For each k-mer type: count, mean position, and the normalised second
    central moment D2 = (1 / (count * n)) * sum_i (pos_i - mean_pos)^2.
    """
    alphabet = "ACGT"
    if k == 1:
        all_kmers = list(alphabet)
    else:
        all_kmers = ["".join(x) for x in iter_product(alphabet, repeat=k)]

    n = len(seq)
    features = []
    for km in all_kmers:
        positions = [i + 1 for i in range(n - len(km) + 1)
                     if seq[i:i + len(km)] == km]
        count = len(positions)
        if count == 0:
            features.extend([0.0, 0.0, 0.0])
        else:
            mean_pos = np.mean(positions)
            d2 = np.sum([(p - mean_pos) ** 2 for p in positions]) / (count * n)
            features.extend([count / n, mean_pos / n, d2])
    return np.array(features)


def nvm_distance_matrix(sequences, k=1):
    """Compute NVM feature vectors and Euclidean distance on standardised features."""
    feat = np.array([nvm_features(s, k) for s in sequences])
    feat_std = StandardScaler().fit_transform(feat)
    return squareform(pdist(feat_std, metric="euclidean")), feat

# version 0: unscaled version (does worse!)
#def nvm_distance_matrix(sequences, k=1):
#    """Compute NVM feature vectors and Euclidean distance matrix."""
#    feat = np.array([nvm_features(s, k) for s in sequences])
#    return squareform(pdist(feat, metric="euclidean")), feat


# ============================================================
# MinHash (Mash-like): Ondov et al., Genome Biology 2016
# ============================================================
def _stable_hash(s):
    return int.from_bytes(hashlib.md5(s.encode()).digest()[:4], "little")


def minhash_signature(seq, k=7, n_hashes=200, seed=42):
    """
    MinHash signature using a deterministic base hash and a fixed family of
    affine permutations h_i(x) = (a_i * x + b_i) mod (2^31 - 1).
    """
    kmers = set(extract_kmers(seq, k))
    if len(kmers) == 0:
        return np.full(n_hashes, np.inf)
    kmer_hashes = np.array([_stable_hash(km) for km in kmers], dtype=np.int64)

    rng = np.random.RandomState(seed)
    large_prime = 2**31 - 1
    a_vals = rng.randint(1, large_prime, size=n_hashes)
    b_vals = rng.randint(0, large_prime, size=n_hashes)

    signature = np.full(n_hashes, np.inf)
    for h in kmer_hashes:
        vals = (a_vals * h + b_vals) % large_prime
        signature = np.minimum(signature, vals)
    return signature


def minhash_distance_matrix(sequences, k=7, n_hashes=200):
    """
    Mash-like MinHash Jaccard distance matrix.
    Jaccard distance = 1 - (matching minhash values / total hashes).
    """
    N = len(sequences)
    sigs = [minhash_signature(s, k, n_hashes) for s in sequences]
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            matches = np.sum(sigs[i] == sigs[j])
            jaccard = matches / n_hashes
            D[i, j] = 1.0 - jaccard
            D[j, i] = D[i, j]
    return D


# ============================================================
# Convenience: compute all baseline distance matrices
# ============================================================

def compute_all_baselines(sequences, k_ffp=3, k_nvm=1, k_mash=7):
    """Compute distance matrices for all baseline methods."""
    results = {}

    print("  Computing FFP-JS distance...")
    results["ffp_js"] = ffp_jensen_shannon_distance(sequences, k=k_ffp)

    print("  Computing FFP-Euclidean distance...")
    results["ffp_euc"] = ffp_euclidean_distance(sequences, k=k_ffp)

    print("  Computing NVM distance...")
    results["nvm"], _ = nvm_distance_matrix(sequences, k=k_nvm)

    print("  Computing MinHash distance...")
    results["minhash"] = minhash_distance_matrix(sequences, k=k_mash)

    return results
