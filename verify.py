"""Verify the ternary circuit theorem. Python 3.10+, NumPy; no optimiser."""
from collections import Counter
from functools import cache
from itertools import combinations, permutations, product
from pathlib import Path
import json

import numpy as np


SCALE = 10**12
ROWS = {
    "3": ["111"],
    "4": ["0012", "1101"],
    "5": ["11001", "00111"],
    "6a": ["100002", "010011", "001101"],
    "6b": ["100002", "010011", "001110"],
    "6c": ["000111", "011001", "101010"],
}
TABLE_UPPER = {"3": -287680000000, "4": -108840000000, "5": -12930000000,
               "6a": -19300000000, "6b": -19300000000, "6c": -2190000000}
COVERED = {3: [1, 1], 4: [0, 6, 1], 5: [0, 15, 25, 1],
           6: [0, 0, 345, 90, 1]}


def ensure(condition, message):
    if not condition:
        raise ValueError(message)


def rref_bases(width):
    """Every row space once, including the zero space."""
    for height in range(width + 1):
        for pivots in combinations(range(width), height):
            free = [(i, j) for i, p in enumerate(pivots)
                    for j in range(p + 1, width) if j not in pivots]
            matrix = np.zeros((height, width), dtype=int)
            for i, p in enumerate(pivots):
                matrix[i, p] = 1
            for digits in product(range(3), repeat=len(free)):
                for (i, j), value in zip(free, digits):
                    matrix[i, j] = value
                yield matrix.copy()


def ln_ratio(a, b):
    """Bounds in units 1/SCALE for log(a/b), 1 <= a/b <= 2."""
    u, v = a - b, a + b
    lower = sum(2 * SCALE * u**j // (j * v**j) for j in range(1, 30, 2))
    # 15 downward roundings, and the tail from j=31 is < 1/SCALE.
    return lower, lower + 16


@cache
def ln_count(n):
    exponent = n.bit_length() - 1
    lo, hi = ln_ratio(n, 1 << exponent)
    lo2, hi2 = ln_ratio(2, 1)
    return exponent * lo2 + lo, exponent * hi2 + hi


@cache
def entropy(histogram):
    """Outward integer endpoints in units of 1/SCALE nat."""
    total = sum(histogram)
    lo, hi = ln_count(total)
    lower_sum = sum(n * ln_count(n)[0] for n in histogram)
    upper_sum = sum(n * ln_count(n)[1] for n in histogram)
    return lo - (upper_sum + total - 1)//total, hi - lower_sum//total


def entropy_of_labels(labels, masses):
    bins = [0] * (int(labels.max()) + 1)
    for label, count in zip(labels, masses):
        bins[label] += count
    return entropy(tuple(sorted(x for x in bins if x)))


def atoms(n):
    values = np.array(list(product(range(3), repeat=n - 1)), dtype=int)
    return np.column_stack((values, -values.sum(axis=1) % 3))


def rows(key):
    return np.array([list(map(int, s)) for s in ROWS[key]])


def laws(key, points, masses):
    n = len(points[0])
    if key == "3":
        support = {(0, 0, 0), (0, 1, 2), (1, 0, 2)}
        return [(1, [int(tuple(v) in support) for v in points])]
    if key in ("6a", "6b"):
        support = {((2*a) % 3, (2*b) % 3, 1+c, 1+d, e, 0)
                   for a,b,c,d,e in product((0,1), repeat=5)
                   if (c+d+e-a-b) % 3 == 1}
        ensure(len(support) == 11, "wrong slice")
        return [(11, [int(tuple(v) in support) for v in points]),
                (18, [1] * len(points))]
    data = masses[str(n)]
    expected = {4: 1000000, 5: 1000002, 6: 1000008}
    ensure(len(data) == len(points) and sum(data) == expected[n]
           and all(type(x) is int and x >= 0 for x in data), "invalid masses")
    return [(1, data)]


def score(components, labels):
    bounds = [0, 0]
    for copies, masses in components:
        lo, hi = entropy_of_labels(labels, masses)
        bounds[0] += copies * lo
        bounds[1] += copies * hi
    return bounds


def gap_bounds(matrix, points, components):
    n = points.shape[1]
    ports = [score(components, col) for col in points.T]
    a_lo, a_hi = map(sum, zip(*ports))
    sig_lo = sig_hi = None
    tests = 0
    for test in rref_bases(n - 1):
        h = len(test)
        contains = not np.any((test @ matrix[:, :-1].T) % 3)
        r = h - 1 + int(contains)
        if r <= 0:
            continue
        values = (points[:, :-1] @ test.T) % 3
        labels = values @ (3**np.arange(h))
        lo, hi = score(components, labels)
        lo, hi = lo//r, (hi+r-1)//r
        sig_lo = lo if sig_lo is None else min(lo, sig_lo)
        sig_hi = hi if sig_hi is None else min(hi, sig_hi)
        tests += 1
    return tests, a_lo - (n-1)*sig_hi, a_hi - (n-1)*sig_lo


def linearly_generated_mask(matrix):
    # Bit i records vector i in the subspace, using its first n-1 digits.
    width = matrix.shape[1] - 1
    coeffs = np.array(list(product(range(3), repeat=len(matrix))), dtype=int)
    ids = ((coeffs @ matrix[:, :-1]) % 3) @ (3**np.arange(width))
    ensure(len(set(ids)) == len(ids), "dependent rows in a claimed basis")
    return sum(1 << int(i) for i in ids)


def verify_coverage(n):
    index_sets = np.array([s for s in product((0,1), repeat=n) if 0 < sum(s) < n])
    seeds = set()
    for key in ROWS:
        if len(ROWS[key][0]) != n:
            continue
        base = rows(key)
        for order in permutations(range(n)):
            seeds.add(linearly_generated_mask(base[:, order]))
    totals = Counter()
    seen = set()
    for basis in rref_bases(n - 1):
        matrix = np.column_stack((basis, -basis.sum(axis=1) % 3))
        candidate = linearly_generated_mask(matrix)
        ensure(candidate not in seen, "repeated row space")
        seen.add(candidate)
        if np.any(np.all((matrix @ index_sets.T) % 3 == 0, axis=0)):
            continue
        ensure(any(candidate & seed == seed for seed in seeds), "uncovered shadow")
        totals[len(matrix)] += 1
    # Gaussian-binomial recurrence counts all subspaces independently of RREF.
    dimension_counts = [1]
    for width in range(1, n):
        dimension_counts = [1] + [
            dimension_counts[d] + 3**(width-d) * dimension_counts[d-1]
            for d in range(1, width)] + [1]
    ensure(len(seen) == sum(dimension_counts), "incomplete subspace enumeration")
    result = [totals[d] for d in range(1, n)]
    ensure(result == COVERED[n], "coverage table does not match")
    return len(seen), result


def main():
    data = json.loads(Path(__file__).with_name("masses.json").read_text())
    for key in ROWS:
        matrix = rows(key)
        n = matrix.shape[1]
        ensure(not np.any(matrix.sum(axis=1) % 3), "rows outside zero-sum space")
        points = atoms(n)
        components = laws(key, points, data)
        tests, lo, hi = gap_bounds(matrix, points, components)
        ensure(hi <= TABLE_UPPER[key] < 0, "claimed table bound failed")
        print(f"{key:>2}  tests={tests:4}  gap in [{lo},{hi}] / {SCALE} nats")
        if n >= 4:
            # Split into the first pair and remaining coordinates; no zero column.
            split = np.zeros((n-2,n), dtype=int)
            split[0,:2] = (1,2)
            for i in range(2,n-1):
                split[i-1,i], split[i-1,-1] = 1,2
            _, control_lo, _ = gap_bounds(split, points, components)
            ensure(control_lo > 0, "split control did not reject the law")
            print(f"    split control gap >= {control_lo} / {SCALE} nats")
    for n in range(3,7):
        total, counts = verify_coverage(n)
        print(f"n={n}  all={total}  nonsplit={sum(counts)}  by dimension={counts}  uncovered=0")


if __name__ == "__main__":
    main()

