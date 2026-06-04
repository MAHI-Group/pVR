"""
pVR: p-Adic Bi-Filtered Simplicial Complexes (parallelised version).
Note: D_H in the code is the compositional L_1 distance on k-mer frequency
vectors, denoted D_c in the paper.

Written by: Tirtharaj Dash (assistance from Claude Opus 4.x), Jan 2026
"""

import numpy as np
from collections import Counter
from itertools import product as iter_product
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb
from gudhi import SimplexTree
from joblib import Parallel, delayed


# Creating simple map here (I think this needs to be done in a more domain-specific way; Later!)
# Otherwise, we should just convey this in the paper.
NUC_MAP = {"A": 1, "C": 2, "G": 3, "T": 4, "U": 4}


def clean_sequence(seq, alphabet="ACGT"):
    return "".join(c for c in seq.upper() if c in alphabet)


def extract_kmers(seq, k):
    if len(seq) < k:
        return []
    return [seq[i:i + k] for i in range(len(seq) - k + 1)]


def kmer_to_padic_int(kmer, p=5, char_map=None):
    if char_map is None:
        char_map = NUC_MAP
    val = 0
    for i, c in enumerate(kmer):
        val += char_map.get(c, 0) * (p ** i)
    return val


def padic_multiscale_histogram(seq, k, p=5, char_map=None, max_scale=None):
    if char_map is None:
        char_map = NUC_MAP
    if max_scale is None:
        max_scale = min(k, 3)
    kmers = extract_kmers(seq, k)
    padic_ints = [kmer_to_padic_int(km, p, char_map) for km in kmers]
    n = max(len(kmers), 1)
    histograms = []
    for j in range(1, max_scale + 1):
        modulus = p ** j
        bins = Counter()
        for val in padic_ints:
            bins[val % modulus] += 1
        hist = np.zeros(modulus)
        for key, count in bins.items():
            if key < modulus:
                hist[key] = count / n
        histograms.append(hist)
    return histograms


def padic_sequence_distance(seq1, seq2, k, p=5, char_map=None, max_scale=None):
    if max_scale is None:
        max_scale = min(k, 3)
    hist1 = padic_multiscale_histogram(seq1, k, p, char_map, max_scale)
    hist2 = padic_multiscale_histogram(seq2, k, p, char_map, max_scale)
    dist = 0.0
    total_weight = 0.0
    for j in range(len(hist1)):
        h1, h2 = hist1[j], hist2[j]
        min_len = min(len(h1), len(h2))
        d = np.sum(np.abs(h1[:min_len] - h2[:min_len]))
        if len(h1) > min_len:
            d += np.sum(np.abs(h1[min_len:]))
        if len(h2) > min_len:
            d += np.sum(np.abs(h2[min_len:]))
        weight = j + 1
        dist += weight * d
        total_weight += weight
    return dist / total_weight


def kmer_frequency_vector(seq, k, alphabet="ACGT"):
    kmers = extract_kmers(seq, k)
    counts = Counter(kmers)
    all_kmers = ["".join(x) for x in iter_product(alphabet, repeat=k)]
    total = max(len(kmers), 1)
    return np.array([counts.get(km, 0) / total for km in all_kmers])


def _padic_row(i, histograms_all, weights):
    N = len(histograms_all)
    row = np.zeros(N)
    total_weight = sum(weights)
    h_i = histograms_all[i]
    for l in range(i + 1, N):
        h_l = histograms_all[l]
        d = 0.0
        for j in range(len(h_i)):
            h1, h2 = h_i[j], h_l[j]
            min_len = min(len(h1), len(h2))
            dd = np.sum(np.abs(h1[:min_len] - h2[:min_len]))
            if len(h1) > min_len:
                dd += np.sum(np.abs(h1[min_len:]))
            if len(h2) > min_len:
                dd += np.sum(np.abs(h2[min_len:]))
            d += weights[j] * dd
        row[l] = d / total_weight
    return i, row


def compute_distance_matrices(sequences, k=4, p=5, char_map=None,
                               alphabet="ACGT", n_jobs=-1):
    N = len(sequences)
    max_scale = min(k, 3)
    histograms = [padic_multiscale_histogram(s, k, p, char_map, max_scale)
                  for s in sequences]
    weights = list(range(1, max_scale + 1))

    print(f"  Computing p-adic distance matrix (n_jobs={n_jobs})...")
    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_padic_row)(i, histograms, weights) for i in range(N)
    )

    D_p = np.zeros((N, N))
    for i, row in results:
        D_p[i, i + 1:] = row[i + 1:]
    D_p = D_p + D_p.T

    print(f"  Computing compositional L_1 distance matrix...")
    freq_vectors = np.array([kmer_frequency_vector(s, k, alphabet)
                             for s in sequences])
    D_H = squareform(pdist(freq_vectors, metric="cityblock"))

    return D_p, D_H, freq_vectors


def make_grid(D_p, D_H, G_p=10, G_h=15):
    vals_p = D_p[np.triu_indices_from(D_p, k=1)]
    vals_p = vals_p[vals_p > 0]
    if len(vals_p) == 0:
        grid_p = np.array([1.0])
    else:
        grid_p = np.quantile(vals_p, np.linspace(0, 1, G_p))
    grid_p = np.unique(grid_p)

    vals_h = D_H[np.triu_indices_from(D_H, k=1)]
    vals_h = vals_h[vals_h > 0]
    if len(vals_h) == 0:
        grid_h = np.array([1.0])
    else:
        grid_h = np.linspace(vals_h.min(), vals_h.max(), G_h)

    return grid_p, grid_h


def compute_betti_at_gridpoint(D_p, D_H, eps_p, eps_h, max_dim=1):
    N = D_p.shape[0]
    st = SimplexTree()
    for i in range(N):
        st.insert([i])
    for i in range(N):
        for j in range(i + 1, N):
            if D_p[i, j] <= eps_p and D_H[i, j] <= eps_h:
                st.insert([i, j])
    st.expansion(max_dim + 1)
    st.compute_persistence()
    betti = st.betti_numbers()
    if isinstance(betti, dict):
        return [betti.get(d, 0) for d in range(max_dim + 1)]
    else:
        return [betti[d] if d < len(betti) else 0 for d in range(max_dim + 1)]


def _gridpoint_work(a, b, eps_p, eps_h, D_p, D_H):
    adj = (D_p <= eps_p) & (D_H <= eps_h)
    np.fill_diagonal(adj, False)
    degrees = adj.sum(axis=1).astype(float)
    betti = compute_betti_at_gridpoint(D_p, D_H, eps_p, eps_h)
    return a, b, degrees, betti


def bifiltration_features(D_p, D_H, grid_p, grid_h, verbose=True, n_jobs=-1):
    N = D_p.shape[0]
    G_p = len(grid_p)
    G_h = len(grid_h)
    degree_profiles = np.zeros((N, G_p, G_h))
    betti_grid = np.zeros((G_p, G_h, 3))

    tasks = [(a, b, ep, eh)
             for a, ep in enumerate(grid_p)
             for b, eh in enumerate(grid_h)]

    if verbose:
        print(f"    Bi-filtration: {len(tasks)} grid points (n_jobs={n_jobs})")

    results = Parallel(n_jobs=n_jobs, backend="threading", verbose=0)(
        delayed(_gridpoint_work)(a, b, ep, eh, D_p, D_H)
        for a, b, ep, eh in tasks
    )

    for a, b, degrees, betti in results:
        degree_profiles[:, a, b] = degrees
        betti_grid[a, b, :len(betti)] = betti[:3]

    deg_flat = degree_profiles.reshape(N, -1)
    betti_flat = betti_grid.reshape(1, -1).repeat(N, axis=0)
    seq_features = np.hstack([deg_flat, betti_flat])

    return seq_features, degree_profiles, betti_grid


def multiscale_padic_features(sequences, k, p=5, char_map=None):
    max_scale = min(k, 3)
    all_features = []
    for seq in sequences:
        hist_list = padic_multiscale_histogram(seq, k, p, char_map, max_scale)
        feats = np.concatenate(hist_list)
        all_features.append(feats)
    return np.array(all_features)


def combined_features(sequences, D_p, D_H, grid_p, grid_h, freq_vectors,
                      k=4, p=5, verbose=True, n_jobs=-1):
    topo_feats, _, betti_grid = bifiltration_features(
        D_p, D_H, grid_p, grid_h, verbose=verbose, n_jobs=n_jobs
    )
    padic_feats = multiscale_padic_features(sequences, k, p)
    all_feats = np.hstack([topo_feats, padic_feats, freq_vectors])
    return all_feats, betti_grid


def padic_only_features(D_p, grid_p):
    N = D_p.shape[0]
    features = np.zeros((N, len(grid_p)))
    for a, eps_p in enumerate(grid_p):
        adj = D_p <= eps_p
        np.fill_diagonal(adj, False)
        features[:, a] = adj.sum(axis=1).astype(float)
    return features


def hamming_only_features(D_H, grid_h):
    N = D_H.shape[0]
    features = np.zeros((N, len(grid_h)))
    for b, eps_h in enumerate(grid_h):
        adj = D_H <= eps_h
        np.fill_diagonal(adj, False)
        features[:, b] = adj.sum(axis=1).astype(float)
    return features


def classify_features(features, labels, method="xgboost", n_folds=5, seed=42):
    """Returns (mean_acc, std_acc, mean_f1, std_f1, fold_accs)."""
    le = LabelEncoder()
    y = le.fit_transform(labels)
    if len(np.unique(y)) < 2:
        return 0.0, 0.0, 0.0, 0.0, []
    min_count = min(Counter(y).values())
    n_folds = min(n_folds, min_count)
    if n_folds < 2:
        n_folds = 2
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scaler = StandardScaler()
    accs, f1s = [], []
    for train_idx, test_idx in skf.split(features, y):
        X_train = scaler.fit_transform(features[train_idx])
        X_test = scaler.transform(features[test_idx])
        y_train_raw, y_test_raw = y[train_idx], y[test_idx]
        fold_le = LabelEncoder()
        y_train = fold_le.fit_transform(y_train_raw)
        valid_test, y_test_mapped = [], []
        for i, yt in enumerate(y_test_raw):
            if yt in fold_le.classes_:
                y_test_mapped.append(fold_le.transform([yt])[0])
                valid_test.append(i)
        if len(valid_test) == 0:
            continue
        y_test = np.array(y_test_mapped)
        X_test = X_test[valid_test]
        if method == "xgboost":
            clf = xgb.XGBClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                eval_metric="mlogloss", verbosity=0,
                random_state=seed, use_label_encoder=False,
                n_jobs=-1,
            )
        elif method == "svm":
            clf = SVC(kernel="rbf", C=10, gamma="scale", random_state=seed)
        elif method == "5nn":
            n = min(5, len(y_train) - 1)
            clf = KNeighborsClassifier(n_neighbors=max(1, n),
                                        metric="euclidean", n_jobs=-1)
        else:
            raise ValueError(f"Unknown method: {method}")
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        accs.append(accuracy_score(y_test, preds))
        f1s.append(f1_score(y_test, preds, average="macro", zero_division=0))
    if len(accs) == 0:
        return 0.0, 0.0, 0.0, 0.0, []
    return float(np.mean(accs)), float(np.std(accs)), \
           float(np.mean(f1s)), float(np.std(f1s)), [float(a) for a in accs]


def classify_distance_matrix(D, labels, k=5, n_folds=5, seed=42):
    """Returns (mean_acc, std_acc, fold_accs)."""
    le = LabelEncoder()
    y = le.fit_transform(labels)
    min_count = min(Counter(y).values())
    n_folds = min(n_folds, min_count)
    if n_folds < 2:
        n_folds = 2
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    accs = []
    for train_idx, test_idx in skf.split(D, y):
        n_neighbors = min(k, len(train_idx) - 1)
        if n_neighbors < 1:
            continue
        clf = KNeighborsClassifier(n_neighbors=n_neighbors, metric="precomputed")
        clf.fit(D[np.ix_(train_idx, train_idx)], y[train_idx])
        preds = clf.predict(D[np.ix_(test_idx, train_idx)])
        accs.append(accuracy_score(y[test_idx], preds))
    if len(accs) == 0:
        return 0.0, 0.0, []
    return float(np.mean(accs)), float(np.std(accs)), [float(a) for a in accs]


def repeated_cv_features(features, labels, method="xgboost",
                         n_folds=5, n_seeds=10, base_seed=42):
    """Repeated stratified k-fold CV. Returns list of n_folds*n_seeds accuracies."""
    all_folds = []
    for s in range(n_seeds):
        _, _, _, _, folds = classify_features(
            features, labels, method=method,
            n_folds=n_folds, seed=base_seed + s,
        )
        all_folds.extend(folds)
    return all_folds


def repeated_cv_distance(D, labels, k=5, n_folds=5, n_seeds=10, base_seed=42):
    """Repeated stratified k-fold CV on a precomputed distance matrix."""
    all_folds = []
    for s in range(n_seeds):
        _, _, folds = classify_distance_matrix(
            D, labels, k=k, n_folds=n_folds, seed=base_seed + s,
        )
        all_folds.extend(folds)
    return all_folds


def merge_small_classes(labels, min_count=3, other_label="Other"):
    counts = Counter(labels)
    return np.array([other_label if counts[l] < min_count else l for l in labels])
