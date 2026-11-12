#!/usr/bin/env python3
"""
EJ Binary Subset Prime Oracle
A simple, explainable prime checker built on the Binary Layer of the Subset System
(from EJ's Base Prime Model & Twin Prime Counterflow Model)

Features:
- Instant modular filters (end digit, mod-30 residue)
- Pure-Python trial division for exact verdict
- Two-layer microscopic neural net trained on binary-subset features
- Human-readable explanations in the language of the model
- Interactive chat mode

Usage:
  python prime_oracle.py              # interactive
  python prime_oracle.py 102197       # single number
  python prime_oracle.py --demo       # run demo tests

Requires only Python 3 + NumPy.
"""

import sys
import numpy as np
from typing import Optional, Tuple, List

# ===================== Core pure-Python primality =====================
def is_prime_pure(n: int) -> bool:
    if n < 2:
        return False
    if n in (2, 3, 5):
        return True
    if n % 2 == 0 or n % 3 == 0 or n % 5 == 0:
        return False
    i = 7
    while i * i <= n:
        if n % i == 0 or n % (i + 4) == 0:
            return False
        i += 6
    return True

# ===================== Binary subset representation =====================
END_DIGIT_TO_BITS = {1: 0b00, 3: 0b01, 7: 0b10, 9: 0b11}
RESIDUES_MOD30 = [1, 7, 11, 13, 17, 19, 23, 29]

def to_subset_binary(n: int) -> Optional[Tuple[int, int, int]]:
    if n <= 5:
        return None
    d = n % 10
    if d not in END_DIGIT_TO_BITS:
        return None
    return n // 10, d, END_DIGIT_TO_BITS[d]

def slot_in_byte(n: int) -> int:
    try:
        return RESIDUES_MOD30.index(n % 30)
    except ValueError:
        return -1

def rise_fall_context(n: int, window: int = 6) -> str:
    start = max(7, n - 300)
    nearby = [c for c in range(start, n + 100, 2) if is_prime_pure(c)]
    digits = [p % 10 for p in nearby]
    if len(digits) < 2:
        return ""
    return "".join("0" if digits[i+1] > digits[i] else "1"
                   for i in range(min(window, len(digits)-1)))

# ===================== Two-layer microscopic model =====================
class MicroPrimeNet:
    def __init__(self):
        self.W1 = self.b1 = self.W2 = self.b2 = None
        self.trained = False

    def extract_features(self, n: int) -> Optional[np.ndarray]:
        rep = to_subset_binary(n)
        if rep is None or slot_in_byte(n) < 0:
            return None
        counter, digit, bits = rep
        slot = slot_in_byte(n)

        # Block density excluding self (no leakage)
        block_start = (n // 30) * 30
        byte = 0
        for i, res in enumerate(RESIDUES_MOD30):
            cand = block_start + res
            if cand != n and is_prime_pure(cand):
                byte |= (1 << i)
        pop = bin(byte).count("1")

        rf = rise_fall_context(n)
        rf_dens = rf.count("1") / max(len(rf), 1)
        four_rises = 1.0 if "0000" in rf else 0.0
        twin = 1.0 if is_prime_pure(n+2) or is_prime_pure(n-2) else 0.0

        # gap residue
        p = n - 1
        while p > 5 and not is_prime_pure(p):
            p -= 1
        gap_res = ((n - p) % 10) / 8.0 if p > 5 else 0.0

        feats = np.array([
            bits / 3.0,
            slot / 7.0,
            byte / 255.0,
            pop / 8.0,
            (counter % 10) / 9.0,
            (counter % 30) / 29.0,
            (n % 7) / 6.0,
            (n % 11) / 10.0,
            (n % 13) / 12.0,
            (n % 210) / 209.0,
            twin,
            gap_res,
            rf_dens,
            four_rises,
            1.0 if n % 6 in (1, 5) else 0.0,
            min(n % 7, 7 - n % 7) / 3.5
        ], dtype=np.float32)
        return feats

    def train(self, max_n: int = 20000, epochs: int = 300):
        samples = list(range(7, max_n))
        X, y = [], []
        for n in samples:
            f = self.extract_features(n)
            if f is not None:
                X.append(f)
                y.append(1.0 if is_prime_pure(n) else 0.0)
        X = np.stack(X)
        y = np.array(y, dtype=np.float32).reshape(-1, 1)

        # simple class balance
        pos = y.sum()
        w = np.where(y == 1, len(y)/(2*pos), len(y)/(2*(len(y)-pos)))

        rng = np.random.default_rng(42)
        self.W1 = rng.normal(0, 0.15, (X.shape[1], 16)).astype(np.float32)
        self.b1 = np.zeros((1, 16), dtype=np.float32)
        self.W2 = rng.normal(0, 0.15, (16, 1)).astype(np.float32)
        self.b2 = np.zeros((1, 1), dtype=np.float32)

        lr = 0.25
        for ep in range(epochs):
            # forward
            h = np.maximum(0, X @ self.W1 + self.b1)          # ReLU
            logits = h @ self.W2 + self.b2
            preds = 1 / (1 + np.exp(-np.clip(logits, -15, 15)))
            # backward
            dlogits = (preds - y) * w
            dW2 = h.T @ dlogits / len(X)
            db2 = dlogits.mean(axis=0, keepdims=True)
            dh = dlogits @ self.W2.T
            dh[h <= 0] = 0
            dW1 = X.T @ dh / len(X)
            db1 = dh.mean(axis=0, keepdims=True)
            # update
            self.W1 -= lr * dW1
            self.b1 -= lr * db1
            self.W2 -= lr * dW2
            self.b2 -= lr * db2
            if ep % 40 == 0:
                lr *= 0.85
        self.trained = True

    def predict_proba(self, n: int) -> float:
        if not self.trained:
            return 0.5
        f = self.extract_features(n)
        if f is None:
            return 0.0
        h = np.maximum(0, f @ self.W1 + self.b1)
        logit = float((h @ self.W2 + self.b2).item())
        return 1 / (1 + np.exp(-np.clip(logit, -15, 15)))

# Global model
_model = MicroPrimeNet()

def explain(n: int) -> str:
    if n < 2:
        return f"{n} is not prime."
    if n in (2, 3, 5):
        return f"{n} is prime."

    rep = to_subset_binary(n)
    if rep is None:
        return f"{n} cannot be prime — end digit {n%10} is forbidden (must be 1, 3, 7 or 9)."
    slot = slot_in_byte(n)
    if slot < 0:
        return f"{n} cannot be prime — not in a residue class coprime to 30."

    counter, digit, bits = rep
    true = is_prime_pure(n)

    if not _model.trained:
        print("Training microscopic model (one-time, ~10-20 s)...", flush=True)
        _model.train()

    score = _model.predict_proba(n)
    lines = [
        f"Number: {n}",
        f"Binary form: 10×{counter} + {digit} (digit bits {bits:02b})",
        f"Mod-30 slot: {slot} / 7",
        f"Exact verdict: {'PRIME' if true else 'COMPOSITE'}",
        f"Micro-model score: {score:.3f}"
    ]
    return "\n".join(lines)

def interactive():
    print("EJ Binary Subset Prime Oracle")
    print("Enter a number (or 'quit'):")
    while True:
        try:
            raw = input("> ").strip()
            if raw.lower() in ("quit", "exit", "q"):
                break
            n = int(raw)
            print(explain(n))
            print()
        except ValueError:
            print("Please enter an integer.")
        except (EOFError, KeyboardInterrupt):
            break

def demo():
    tests = [11, 15, 17, 21, 101, 10007, 102197, 999983, 1000003, 49, 121]
    for t in tests:
        print(explain(t))
        print("-" * 40)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        interactive()
    elif sys.argv[1] == "--demo":
        demo()
    else:
        try:
            print(explain(int(sys.argv[1])))
        except ValueError:
            print("Usage: python prime_oracle.py [number | --demo]")
