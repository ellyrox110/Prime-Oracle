#!/usr/bin/env python3
"""
Simple Prime Oracle LLM (dependency-free)
Built from the Binary Layer of the Subset System

- Pure-Python trial division (no sympy)
- Microscopic model (tiny logistic regression / neural net in NumPy)
  trained on binary features from the subset system
"""

import numpy as np
from typing import Tuple, Optional, List

# ---------- Pure-Python primality (trial division) ----------
def is_prime_pure(n: int) -> bool:
    """Dependency-free primality test by trial division."""
    if n < 2:
        return False
    if n in (2, 3, 5):
        return True
    if n % 2 == 0 or n % 3 == 0 or n % 5 == 0:
        return False
    i = 7
    while i * i <= n:
        if n % i == 0 or n % (i + 4) == 0:  # wheel of 6
            return False
        i += 6
    return True

# ---------- Ask 1: Binary representation ----------
END_DIGIT_TO_BITS = {1: 0b00, 3: 0b01, 7: 0b10, 9: 0b11}

def to_subset_binary(n: int) -> Optional[Tuple[int, int, int]]:
    if n <= 5:
        return None
    d = n % 10
    if d not in END_DIGIT_TO_BITS:
        return None
    counter = n // 10
    bits = END_DIGIT_TO_BITS[d]
    return counter, d, bits

# ---------- Ask 2: Subset byte (mod 30) ----------
RESIDUES_MOD30 = [1, 7, 11, 13, 17, 19, 23, 29]

def slot_in_byte(n: int) -> int:
    r = n % 30
    try:
        return RESIDUES_MOD30.index(r)
    except ValueError:
        return -1

def subset_byte_for(n: int, prime_fn=is_prime_pure) -> int:
    """Compute the 8-bit subset byte using the pure primality test."""
    block_start = (n // 30) * 30
    byte = 0
    for i, res in enumerate(RESIDUES_MOD30):
        candidate = block_start + res
        if prime_fn(candidate):
            byte |= (1 << i)
    return byte

# ---------- Ask 3: Rise/fall skeleton ----------
def rise_fall_context(n: int, window: int = 5) -> str:
    """Simple local rise/fall (0=rise, 1=fall) using pure search for nearby primes."""
    # Generate a short list of candidates around n and filter
    start = max(7, n - 200)
    candidates = list(range(start, n + 200, 2))  # odds
    nearby = [c for c in candidates if is_prime_pure(c)]
    digits = [p % 10 for p in nearby]
    if len(digits) < 2:
        return "insufficient"
    bits = []
    for i in range(min(window, len(digits)-1)):
        bits.append("0" if digits[i+1] > digits[i] else "1")
    return "".join(bits)

# ---------- Microscopic model (strengthened) ----------
class MicroPrimeModel:
    """
    Strengthened microscopic model trained on richer binary-subset features.
    Features (~14):
      - end-digit bits
      - slot in byte
      - subset-byte value + popcount + bit for this slot
      - counter modular residues (10, 30, 7, 11)
      - twin-flag (n+2 or n-2 prime)
      - gap residue to previous prime (mod 10, normalized)
      - longer rise/fall density + presence of 0000 pattern
      - distance to nearest multiple of small primes (normalized)
    """
    def __init__(self):
        self.weights = None
        self.bias = 0.0
        self.trained = False
        self.feature_count = 14

    def _nearest_prime_gap_residue(self, n: int) -> float:
        """Gap residue (mod 10) to the previous prime, normalized 0-1."""
        if n <= 7:
            return 0.0
        p = n - 1
        while p > 5 and not is_prime_pure(p):
            p -= 1
        if p <= 5:
            return 0.0
        gap = n - p
        return (gap % 10) / 8.0  # possible even residues 0,2,4,6,8

    def _is_twin(self, n: int) -> float:
        """1.0 if part of a twin pair, else 0.0."""
        return 1.0 if (is_prime_pure(n + 2) or is_prime_pure(n - 2)) else 0.0

    def extract_features(self, n: int) -> Optional[np.ndarray]:
        rep = to_subset_binary(n)
        if rep is None:
            return None
        counter, digit, bits = rep
        slot = slot_in_byte(n)
        if slot == -1:
            return None

        # Compute subset byte *excluding* n itself to avoid leakage
        block_start = (n // 30) * 30
        byte = 0
        for i, res in enumerate(RESIDUES_MOD30):
            candidate = block_start + res
            if candidate != n and is_prime_pure(candidate):
                byte |= (1 << i)
        popcount = bin(byte).count("1")

        # Longer rise/fall (uses nearby primes, not n)
        rf = rise_fall_context(n, window=8)
        rf_density = rf.count("1") / max(len(rf), 1)
        has_four_rises = 1.0 if "0000" in rf else 0.0

        # Twin flag (n+2 or n-2) – this is a real signal, not leakage of n itself
        twin = self._is_twin(n)
        gap_res = self._nearest_prime_gap_residue(n)

        # Extra modular
        mod7 = (n % 7) / 6.0
        mod11 = (n % 11) / 10.0
        dist7 = min(n % 7, 7 - (n % 7)) / 3.5
        dist11 = min(n % 11, 11 - (n % 11)) / 5.5

        feats = np.array([
            bits / 3.0,                  # 0 end-digit bits
            slot / 7.0,                  # 1 slot
            byte / 255.0,                # 2 other primes in block
            popcount / 8.0,              # 3 density of block
            (counter % 10) / 9.0,        # 4
            (counter % 30) / 29.0,       # 5
            mod7,                        # 6
            mod11,                       # 7
            twin,                        # 8 twin flag
            gap_res,                     # 9 gap residue
            rf_density,                  # 10
            has_four_rises,              # 11
            (dist7 + dist11) / 2.0,      # 12
            (n % 210) / 209.0,           # 13 mod 210
            1.0 if n % 6 in (1, 5) else 0.0,  # 14 admissible mod 6
            (n % 13) / 12.0              # 15 mod 13
        ], dtype=np.float32)
        return feats

    def train(self, samples: List[int], epochs: int = 400, lr: float = 0.2):
        """Train a single-layer logistic regression on the samples with class balancing."""
        X, y = [], []
        for n in samples:
            f = self.extract_features(n)
            if f is not None:
                X.append(f)
                y.append(1.0 if is_prime_pure(n) else 0.0)
        if not X:
            return
        X = np.stack(X)
        y = np.array(y, dtype=np.float32)
        # Class weights
        n_pos = np.sum(y)
        n_neg = len(y) - n_pos
        w_pos = len(y) / (2 * n_pos) if n_pos > 0 else 1.0
        w_neg = len(y) / (2 * n_neg) if n_neg > 0 else 1.0
        sample_weights = np.where(y == 1, w_pos, w_neg)

        self.weights = np.random.randn(X.shape[1]) * 0.03
        self.bias = 0.0
        for epoch in range(epochs):
            logits = X @ self.weights + self.bias
            preds = 1 / (1 + np.exp(-np.clip(logits, -20, 20)))
            error = (preds - y) * sample_weights
            grad_w = X.T @ error / len(X)
            grad_b = np.mean(error)
            lr_now = lr * (0.997 ** epoch)
            self.weights -= lr_now * grad_w
            self.bias -= lr_now * grad_b
        self.trained = True

    def predict_proba(self, n: int) -> float:
        if not self.trained:
            return 0.5
        f = self.extract_features(n)
        if f is None:
            return 0.0
        logit = float(f @ self.weights + self.bias)
        return 1 / (1 + np.exp(-np.clip(logit, -20, 20)))

    def predict(self, n: int, threshold: float = 0.5) -> bool:
        return self.predict_proba(n) >= threshold

# ---------- Ask 4: The simple LLM interface ----------
# Global microscopic model (trained on first call)
_model = MicroPrimeModel()

def is_prime_llm(n: int, explain: bool = True, use_model: bool = True) -> str:
    """
    Simple LLM-style oracle (dependency-free).
    Binary layers + pure trial division for certainty.
    Optional microscopic model for a learned score.
    """
    if n < 2:
        return f"{n} is not prime (too small)."
    if n in (2, 3, 5):
        return f"{n} is prime (special small case)."

    # Ask 1
    rep = to_subset_binary(n)
    if rep is None:
        return f"{n} cannot be prime: end digit {n%10} is forbidden (must be 1, 3, 7 or 9)."

    counter, digit, bits = rep

    # Ask 2
    slot = slot_in_byte(n)
    if slot == -1:
        return f"{n} cannot be prime: not in a residue class coprime to 30."

    byte = subset_byte_for(n)
    bit_set = bool(byte & (1 << slot))

    # Ask 3
    rf = rise_fall_context(n)

    # Ground truth
    actually_prime = is_prime_pure(n)

    # Microscopic model score
    if use_model and not _model.trained:
        # Larger training set for the strengthened model
        train_set = list(range(7, 30000, 1))
        _model.train(train_set, epochs=350, lr=0.15)
    model_score = _model.predict_proba(n) if use_model else None

    if explain:
        lines = [
            f"Number: {n}",
            f"Ask 1 – Binary form: counter={counter}, digit={digit} (bits {bits:02b}) → {10*counter + digit}",
            f"Ask 2 – Subset byte of its 30-block: 0x{byte:02X}; slot {slot} is {'SET' if bit_set else 'CLEAR'}",
            f"Ask 3 – Local rise/fall skeleton: {rf}",
            f"Ask 4 – Verdict (trial division): {'PRIME' if actually_prime else 'COMPOSITE'}"
        ]
        if model_score is not None:
            lines.append(f"         Micro-model score: {model_score:.3f} ({'prime' if model_score >= 0.5 else 'composite'})")
        return "\n".join(lines)
    else:
        return "prime" if actually_prime else "composite"


# ---------- Demo ----------
if __name__ == "__main__":
    tests = [11, 13, 15, 17, 19, 21, 29, 31, 100, 101, 102197, 999983, 10007, 10009]
    for t in tests:
        print(is_prime_llm(t))
        print("-" * 50)
