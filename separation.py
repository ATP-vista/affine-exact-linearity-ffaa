"""Exact check of the explicit separations in Section 4.4 (Proposition 4.5 and the closing remark).

Python 3.10+, standard library only.  All inequalities are decided in exact rational arithmetic.
Logarithms enter only through rational upper bounds L for ln p and ln 2, each certified by
exp(L) > p using a truncated exponential series, which is a lower bound for exp(L) when L > 0.
"""
from fractions import Fraction
from itertools import product
import cmath, math


def exp_lower(x: Fraction, terms: int = 60) -> Fraction:
    """A lower bound for exp(x), x > 0: the partial sum of its power series."""
    total, term = Fraction(0), Fraction(1)
    for i in range(terms):
        total += term
        term = term * x / (i + 1)
    return total


def certified_log_upper(n: int, L: Fraction) -> Fraction:
    assert exp_lower(L) > n, (n, L)
    return L


def signed_relations(p, c):
    return [u for u in product((-1, 0, 1), repeat=len(c)) if sum(a * b for a, b in zip(u, c)) % p == 0]


def check(p, c, N, a, eps, lam):
    m = len(c)
    k = m - 2
    # 1. Only constant signed relations, so A(s) = 1 + 2 eps^m at every input (weights 1).
    rels = signed_relations(p, c)
    assert sorted(rels) == sorted([(0,) * m, (1,) * m, (-1,) * m]), rels
    # Floating-point spot check of A(s) on one column for the explicit chart when k <= 1.
    if k == 1:
        pts = [1, (-c[0] * pow(c[1], -1, p)) % p, 0]          # carries c[0], c[1], c[2]
        assert (sum(ci * x for ci, x in zip(c, pts))) % p == 0
        f = lambda i, y: 1 + 2 * float(eps) * math.cos(2 * math.pi * c[i] * y / p)
        for s in range(p):
            A = sum(math.prod(f(i, (z - pts[i] * s) % p) for i in range(m)) for z in range(p)) / p
            assert abs(A - (1 + 2 * float(eps) ** m)) < 1e-12
    # 2. Inclusion probabilities: M (F/p)^N <= bound, F = 1 + 2 eps, M = p^a / lam.
    M = Fraction(p) ** a / lam
    F = 1 + 2 * eps
    incl = M * (F / p) ** N
    # 3. Expected admissible offsets per input and the union bound over p^{kN} inputs.
    mu = M ** m * Fraction(1, p ** ((m - 1) * N)) * (1 + 2 * eps ** m) ** N
    lnp = certified_log_upper(p, {7: Fraction(1946, 1000), 31: Fraction(3434, 1000)}[p])
    ln2 = certified_log_upper(2, Fraction(6932, 10000))
    union_ok = mu > k * N * lnp + ln2            # => p^{kN} e^{-mu} < 1/2
    # 4. Chebyshev: Pr[|R_x| > lam M] <= Var/((lam-1)M)^2 <= 1/((lam-1)^2 M); m ports.
    cheb = Fraction(m) / ((lam - 1) ** 2 * M)
    # 5. Linear optimum: sum_x min(A_x, N) >= (m-1) N forces max A_x >= ceil((m-1)N/m).
    linear_opt = -(-(m - 1) * N // m)
    assert incl <= 1 and union_ok and cheb < Fraction(1, 2) and a < linear_opt
    return incl, mu, k * N * lnp + ln2, cheb, linear_opt


if __name__ == "__main__":
    cases = [
        ("Proposition 4.5: points 0,1,3 of F_7, relation (1,2,4) on (1,3,0)", 7, (1, 2, 4), 38, 25,
         Fraction(12, 25), Fraction(3, 2)),
        ("Remark after Proposition 4.5: F_31, relation (1,2,4,8,16)", 31, (1, 2, 4, 8, 16), 244, 195,
         Fraction(49, 100), Fraction(3, 2)),
    ]
    for name, p, c, N, a, eps, lam in cases:
        incl, mu, thresh, cheb, lin = check(p, c, N, a, eps, lam)
        print(name)
        print(f"  N={N}, budget {a} per port; linear codes need {lin}")
        print(f"  max inclusion probability {float(incl):.6f} (<= 1)")
        print(f"  expected admissible offsets per input mu = {float(mu):.6f}; certified kN ln p + ln 2 < {float(thresh):.4f}")
        print(f"  Chebyshev bound for oversized alphabets {float(cheb):.3e} (< 1/2)")
    # Failing control: one block shorter (N=37, budget 24 is still below the linear optimum 25) must fail.
    try:
        check(7, (1, 2, 4), 37, 24, Fraction(12, 25), Fraction(3, 2))
    except AssertionError:
        print("control: N=37 with budget 24 is rejected, as expected")
    else:
        raise SystemExit("control unexpectedly passed")
    print("all checks passed")
