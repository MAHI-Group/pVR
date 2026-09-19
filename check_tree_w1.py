"""
Numerical check of the tree-Wasserstein identities for D_p and D_c.

Solves the transport problem exactly (linear programme) on random sequences
and compares with the distances as implemented in pvr_eff.py. Also checks the
bound W1(wHam) / W1(p-adic) in [1, p/(p-1)] for Reviewer 3's weighted Hamming
metric.

Usage:
  python check_tree_w1.py
"""

import itertools

import numpy as np
from scipy.optimize import linprog

from pvr_eff import kmer_frequency_vector, padic_sequence_distance

K, P = 4, 5
KMERS = ["".join(x) for x in itertools.product("ACGT", repeat=K)]


def lcp(u, v):
    j = 0
    while j < len(u) and u[j] == v[j]:
        j += 1
    return j


LCP = np.array([[lcp(u, v) for v in KMERS] for u in KMERS])
OFF = LCP < K


def cost_from_f(f):
    C = np.zeros(LCP.shape)
    C[OFF] = np.asarray(f)[LCP[OFF]]
    return C


def w1(mu, nu, C):
    n = len(mu)
    rows = np.kron(np.eye(n), np.ones(n))
    cols = np.kron(np.ones(n), np.eye(n))
    res = linprog(C.ravel(), A_eq=np.vstack([rows, cols]),
                  b_eq=np.concatenate([mu, nu]), bounds=(0, None), method="highs")
    return res.fun


def random_seq(rng, n):
    return "".join(rng.choice(list("ACGT"), size=n, p=rng.dirichlet(np.ones(4) * 2)))


def main():
    rng = np.random.default_rng(0)
    C_impl = cost_from_f([2.0, 5.0 / 3.0, 1.0, 0.0])
    C_disc = cost_from_f([2.0] * K)
    C_padic = cost_from_f([P ** -j for j in range(K)])
    C_wham = np.array([[sum(P ** -i for i in range(K) if u[i] != v[i]) for v in KMERS]
                       for u in KMERS])

    print(f"{'D_p':>9s} {'W1 impl':>9s} {'D_c':>9s} {'W1 disc':>9s} "
          f"{'W1 padic':>9s} {'W1 wHam':>9s} {'ratio':>7s}")
    worst = 0.0
    for _ in range(8):
        s, t = random_seq(rng, rng.integers(300, 1500)), random_seq(rng, rng.integers(300, 1500))
        mu, nu = kmer_frequency_vector(s, K), kmer_frequency_vector(t, K)
        dp, dc = padic_sequence_distance(s, t, K, p=P), np.abs(mu - nu).sum()
        wi, wd = w1(mu, nu, C_impl), w1(mu, nu, C_disc)
        wp, wh = w1(mu, nu, C_padic), w1(mu, nu, C_wham)
        worst = max(worst, abs(dp - wi), abs(dc - wd))
        print(f"{dp:9.6f} {wi:9.6f} {dc:9.6f} {wd:9.6f} {wp:9.6f} {wh:9.6f} {wh / wp:7.4f}")
    print(f"max identity gap {worst:.2e}; ratio bound p/(p-1) = {P / (P - 1):.4f}")


if __name__ == "__main__":
    main()
