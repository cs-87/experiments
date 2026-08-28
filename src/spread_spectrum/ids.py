"""
The set of valid 32-bit watermark IDs, and the arithmetic for choosing it well.

Why this file exists at all. The detector scores its 32-dimensional evidence against
every valid ID and picks the best. Two IDs that differ in d bit positions produce
scores that differ by 2 * sum over those d positions of the per-bit evidence, so under
Gaussian evidence with per-bit mean mu and standard deviation sigma the chance of
confusing them is Q(mu * sqrt(d) / sigma). The minimum Hamming distance of the ID set
therefore sets the error exponent directly, and it is a property of the ID set alone --
it costs nothing in embedded energy and nothing in perceptual distortion.

200,000 IDs drawn at random from the 2^32 available have minimum distance 1: there are
about 154 colliding pairs at distance 1 and 2,463 at distance 2, and 85% of IDs have a
neighbour within distance 4. Choosing the IDs as an error-correcting code instead buys
sqrt(d_min) in effective amplitude -- 7.8 dB at d=6, 9.0 dB at d=8 -- which is the
largest single lever in the whole system and the only free one.

A linear code buys a second thing that matters operationally: its distance distribution
is identical around every codeword, so every customer gets the same misattribution rate.
With random IDs the error probability varies by ID and whoever draws an unlucky one is
permanently worse off.

Constructions here:
  extended_bch_32_21   [32, 21, 6]  2,097,152 codewords  -- the recommended source set
  reed_muller_2_5      [32, 16, 8]     65,536 codewords  -- d=8, but short of 200k
  best_subcode(k)      searched k-dimensional subcodes of the above
  random_ids(n)        the baseline to beat
"""

import argparse
import itertools

import numpy as np

BIT_LENGTH = 32


# --------------------------------------------------------------------------------
# GF(2) polynomial helpers and GF(32)
# --------------------------------------------------------------------------------

GF32_PRIMITIVE = 0b100101          # x^5 + x^2 + 1


def _gf_tables(prim=GF32_PRIMITIVE, m=5):
    """exp[i] = alpha^i as a 5-bit integer; log is its inverse."""
    n = (1 << m) - 1
    exp = np.zeros(2 * n, dtype=np.int64)
    log = np.zeros(1 << m, dtype=np.int64)
    x = 1
    for i in range(n):
        exp[i] = x
        log[x] = i
        x <<= 1
        if x & (1 << m):
            x ^= prim
    exp[n:] = exp[:n]
    return exp, log


def _poly_mul_gf2(a, b):
    """Multiply two GF(2) polynomials given as bit-masks (bit i = coefficient of x^i)."""
    out = 0
    while b:
        if b & 1:
            out ^= a
        a <<= 1
        b >>= 1
    return out


def _minimal_polynomial(power, exp, log, m=5):
    """
    Minimal polynomial over GF(2) of alpha^power, as a bit-mask.

    Product over the conjugacy class {power, 2*power, 4*power, ...} of (x + alpha^j),
    expanded with GF(32) coefficients; the result is guaranteed to have coefficients in
    GF(2), which is asserted rather than assumed.
    """
    n = (1 << m) - 1
    conj, p = [], power % n
    while p not in conj:
        conj.append(p)
        p = (2 * p) % n

    # coefficients over GF(32), lowest degree first, starting from the constant poly 1
    coeffs = [1]
    for j in conj:
        root = exp[j]
        new = [0] * (len(coeffs) + 1)
        for i, c in enumerate(coeffs):
            new[i + 1] ^= c                                    # c * x
            if c and root:
                new[i] ^= exp[(log[c] + log[root]) % n]        # c * alpha^j
        coeffs = new

    mask = 0
    for i, c in enumerate(coeffs):
        assert c in (0, 1), f"minimal polynomial left GF(2): coefficient {c}"
        mask |= c << i
    return mask


def _bch_generator(designed_distance=5, m=5):
    """Generator polynomial of the primitive narrow-sense BCH code of length 2^m - 1."""
    exp, log = _gf_tables(m=m)
    g, seen = 1, set()
    n = (1 << m) - 1
    for power in range(1, designed_distance):
        p, cls = power % n, set()
        while p not in cls:
            cls.add(p)
            p = (2 * p) % n
        if cls & seen:
            continue
        seen |= cls
        g = _poly_mul_gf2(g, _minimal_polynomial(power, exp, log, m=m))
    return g


# --------------------------------------------------------------------------------
# Codeword enumeration
# --------------------------------------------------------------------------------

def _span(generators):
    """
    All 2^k XOR-combinations of `generators`, as uint32.

    Doubling rather than iterating over 2^k index tuples: 2^21 codewords come out in
    21 vectorised concatenations instead of two million Python-level loop steps.
    """
    words = np.zeros(1, dtype=np.uint32)
    for g in generators:
        words = np.concatenate([words, words ^ np.uint32(g)])
    return words


def _weights(words):
    return np.bitwise_count(words.astype(np.uint32)).astype(np.int64)


def _rank_gf2(rows):
    rows, rank = list(rows), 0
    for bit in range(BIT_LENGTH):
        pivot = next((i for i, r in enumerate(rows) if r >> bit & 1), None)
        if pivot is None:
            continue
        p = rows.pop(pivot)
        rows = [r ^ p if r >> bit & 1 else r for r in rows]
        rank += 1
    return rank


# --------------------------------------------------------------------------------
# The constructions
# --------------------------------------------------------------------------------

def extended_bch_32_21():
    """
    [32, 21, 6] generators as uint32.

    BCH(31, 21) with designed distance 5, extended by an overall parity bit. The parity
    bit is what lifts the distance from 5 to 6: every odd-weight codeword gains a one,
    so all weights become even and weight 5 becomes weight 6.
    """
    g = _bch_generator(designed_distance=5)
    rows = []
    for i in range(21):
        w = g << i                                       # x^i * g(x), a length-31 word
        w = (w << 1) | (int(np.bitwise_count(np.uint32(w))) & 1)   # overall parity, LSB
        rows.append(w)
    return rows


def reed_muller_2_5():
    """[32, 16, 8] generators as uint32: RM(2, 5), monomials of degree <= 2 in 5 vars."""
    idx = np.arange(32, dtype=np.uint32)
    x = [(idx >> b) & 1 for b in range(5)]

    def word(bits):
        return int(np.bitwise_or.reduce(bits.astype(np.uint32) << idx))

    rows = [word(np.ones(32, dtype=np.uint32))]
    rows += [word(x[i]) for i in range(5)]
    rows += [word(x[i] & x[j]) for i, j in itertools.combinations(range(5), 2)]
    return rows


def random_ids(n, seed=0):
    """n distinct uniform 32-bit IDs. The baseline a designed codebook has to beat."""
    rng = np.random.default_rng(seed)
    out = np.array([], dtype=np.uint32)
    while out.size < n:
        draw = rng.integers(0, 1 << 32, n - out.size + 1024, dtype=np.uint64).astype(np.uint32)
        out = np.unique(np.concatenate([out, draw]))
    return np.sort(rng.permutation(out)[:n])


# --------------------------------------------------------------------------------
# Distance analysis
# --------------------------------------------------------------------------------

def linear_code_stats(generators):
    """Exact min distance and weight distribution, via the linearity of the code."""
    words = _span(generators)
    w = _weights(words)
    counts = np.bincount(w, minlength=BIT_LENGTH + 1)
    nonzero = w[w > 0]
    return {
        "k": len(generators),
        "size": int(words.size),
        "min_distance": int(nonzero.min()),
        "weight_distribution": {int(i): int(c) for i, c in enumerate(counts) if c},
        "words": words,
    }


def nonlinear_min_distance(ids, max_d=3):
    """
    Smallest pairwise Hamming distance in an arbitrary ID set, up to `max_d`.

    Pairwise comparison over 200k IDs is 2e10 pairs. Instead, for each error pattern of
    weight <= max_d, XOR it into the whole set and ask whether the result is a member --
    5,489 sorted lookups over 200k elements rather than 2e10 comparisons. Returns
    max_d + 1 to mean "no pair closer than max_d", not a measured value.
    """
    ids = np.sort(np.asarray(ids, dtype=np.uint32))
    found = {}
    for d in range(1, max_d + 1):
        hits = 0
        for positions in itertools.combinations(range(BIT_LENGTH), d):
            pattern = np.uint32(sum(1 << p for p in positions))
            other = ids ^ pattern
            pos = np.searchsorted(ids, other)
            pos[pos >= ids.size] = 0
            hits += int((ids[pos] == other).sum())
        found[d] = hits // 2                     # each pair is seen from both ends
        if found[d]:
            break
    md = next((d for d in sorted(found) if found[d]), max_d + 1)
    return md, found


def union_bound(weight_distribution, z, size=None):
    """
    Union bound on codeword error for soft ML decoding at per-bit SNR `z = mu / sigma`.

    A codeword at Hamming distance d is preferred to the true one when the noise on
    those d bits outweighs 2*d*mu, which happens with probability Q(z * sqrt(d)). For a
    linear code the weight distribution is the distance distribution seen from every
    codeword, so this is the same bound for all of them.
    """
    from math import erfc, sqrt
    q = lambda x: 0.5 * erfc(x / sqrt(2))
    total = 0.0
    for d, count in weight_distribution.items():
        if d == 0:
            continue
        if size is not None:
            count = count * size / (2 ** int(np.log2(sum(weight_distribution.values()))))
        total += count * q(z * sqrt(d))
    return min(total, 1.0)


# --------------------------------------------------------------------------------
# Subcode search
# --------------------------------------------------------------------------------

def best_subcode(generators, k, trials=400, seed=0, progress=False):
    """
    Search k-dimensional subcodes of the span of `generators` for the largest minimum
    distance.

    A subcode inherits its parent's minimum distance as a lower bound and often beats
    it, because the low-weight codewords may all lie outside the chosen subspace. The
    search is random over full-rank k x len(generators) binary combination matrices; the
    space is far too large to enumerate and the payoff is a plateau, not a needle.
    """
    rng = np.random.default_rng(seed)
    best = None
    for t in range(trials):
        mix = rng.integers(0, 2, (k, len(generators)))
        rows = [int(np.bitwise_xor.reduce(
            np.array([g for g, m in zip(generators, row) if m] or [0], dtype=np.uint32)))
            for row in mix]
        if _rank_gf2(rows) != k:
            continue
        w = _weights(_span(rows))
        md = int(w[w > 0].min())
        if best is None or md > best[0]:
            best = (md, rows)
            if progress:
                print(f"  trial {t}: d_min = {md}")
    return best


# --------------------------------------------------------------------------------
# The public object
# --------------------------------------------------------------------------------

class CodewordSet:
    """
    A set of valid 32-bit IDs plus the ±1 matrix the decoder multiplies against.

    `signs` is (N, 32) with column 0 the most significant bit, matching
    utils.bit.get_bit_string and therefore matching the PRN index used by
    prn.BalancedPRNGenerator.combine.
    """

    def __init__(self, ids, name="", min_distance=None, weight_distribution=None):
        self.ids = np.sort(np.asarray(ids, dtype=np.uint32))
        self.name = name
        self.min_distance = min_distance
        self.weight_distribution = weight_distribution

    def __len__(self):
        return int(self.ids.size)

    @property
    def signs(self):
        bits = (self.ids[:, None] >> np.arange(BIT_LENGTH - 1, -1, -1, dtype=np.uint32)) & 1
        return np.where(bits == 1, 1, -1).astype(np.int8)

    def contains(self, value):
        pos = np.searchsorted(self.ids, np.uint32(value))
        return bool(pos < self.ids.size and self.ids[pos] == np.uint32(value))

    def save(self, path):
        np.save(path, self.ids)

    @classmethod
    def load(cls, path, name=""):
        return cls(np.load(path), name=name)

    @classmethod
    def from_linear(cls, generators, name="", count=None, seed=0):
        stats = linear_code_stats(generators)
        ids = stats["words"]
        if count is not None and count < ids.size:
            # A subset of a linear code keeps the parent's minimum distance, but its
            # distance distribution is no longer uniform across members. Taking the
            # lowest `count` by value is deterministic and reproducible; the weight
            # distribution reported stays the parent's, as the guaranteed bound.
            ids = np.sort(ids)[:count]
        return cls(ids, name=name, min_distance=stats["min_distance"],
                   weight_distribution=stats["weight_distribution"])

    @classmethod
    def recommended(cls, count=200_000):
        """[32, 21, 6] extended BCH, truncated to `count`. d_min = 6 guaranteed."""
        return cls.from_linear(extended_bch_32_21(), name=f"extBCH[32,21,6]/{count}",
                               count=count)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Codeword set construction and analysis")
    ap.add_argument("--count", type=int, default=200_000)
    ap.add_argument("--subcode-trials", type=int, default=300)
    ap.add_argument("--save", default=None, help="write the recommended ID set to this .npy")
    args = ap.parse_args(argv)

    print("=" * 78)
    print("LINEAR CONSTRUCTIONS")
    print("=" * 78)
    for name, gens in [("extended BCH [32,21,*]", extended_bch_32_21()),
                       ("Reed-Muller RM(2,5) [32,16,*]", reed_muller_2_5())]:
        s = linear_code_stats(gens)
        wd = {d: c for d, c in s["weight_distribution"].items() if d}
        print(f"\n{name}: k={s['k']} size={s['size']:,} d_min={s['min_distance']}")
        print("  weight distribution:",
              " ".join(f"{d}:{c:,}" for d, c in sorted(wd.items())[:8]),
              "..." if len(wd) > 8 else "")
        for z in (2.0, 2.5, 3.0):
            print(f"  union-bound codeword error at per-bit z={z}: "
                  f"{union_bound(s['weight_distribution'], z):.3e}")

    print("\n" + "=" * 78)
    print(f"SUBCODE SEARCH (need >= {args.count:,} codewords)")
    print("=" * 78)
    gens = extended_bch_32_21()
    for k in (18, 19, 20):
        if 2 ** k < args.count:
            continue
        best = best_subcode(gens, k, trials=args.subcode_trials)
        print(f"  best [{BIT_LENGTH},{k}] subcode found over {args.subcode_trials} trials: "
              f"d_min={best[0]}  size={2**k:,}")

    print("\n" + "=" * 78)
    print(f"RANDOM {args.count:,} IDs -- the baseline")
    print("=" * 78)
    rnd = random_ids(args.count, seed=0)
    md, hits = nonlinear_min_distance(rnd, max_d=3)
    print(f"  d_min = {md}   colliding pairs by distance: "
          + "  ".join(f"d={d}: {n:,}" for d, n in sorted(hits.items())))

    print("\n" + "=" * 78)
    print("RECOMMENDED SET")
    print("=" * 78)
    cs = CodewordSet.recommended(args.count)
    # No exhaustive check here: linearity already proves it. The difference of two
    # codewords is itself a codeword, so the distance between any two members of ANY
    # subset is the weight of some nonzero codeword and is therefore at least d_min.
    # Searching weight <= 5 patterns over 200k IDs would be 2.4e5 patterns x 2e5
    # lookups spent re-deriving a theorem.
    md, hits = nonlinear_min_distance(cs.ids, max_d=2)
    print(f"  {cs.name}: {len(cs):,} IDs, d_min={cs.min_distance} "
          f"(guaranteed by linearity for every subset); "
          f"spot check at distance <= 2 found {sum(hits.values())} pairs")
    print(f"  signs matrix: {cs.signs.shape} {cs.signs.dtype} "
          f"({cs.signs.nbytes / 1e6:.1f} MB); a decode is one "
          f"{len(cs):,}x32 GEMM = {2 * len(cs) * 32 / 1e6:.1f} MFLOP")
    if args.save:
        cs.save(args.save)
        print(f"  saved to {args.save}")


if __name__ == "__main__":
    main()
