#!/usr/bin/env python3
"""
Binary Layer of the Subset System
Amended prototype matching the end-digit / subset binary description.

A prime > 5 is represented as:
    prime = 10 * (base + counter) + digit

where:
- digit ∈ {1, 3, 7, 9} encoded in 2 bits
- counter is a binary counter that steps across blocks of 30

One byte (8 bits) covers one block of 30 consecutive integers,
tracking the 8 residues coprime to 30:
    1, 7, 11, 13, 17, 19, 23, 29

The rise/fall skeleton operates on sequences of these 2-bit digits
and forbids patterns containing four consecutive rises (0000 in the
rise/fall encoding).
"""

import sympy
from typing import List, Tuple, Dict
from collections import Counter
import itertools

# The 8 admissible residues mod 30 (coprime to 30)
RESIDUES_MOD30 = [1, 7, 11, 13, 17, 19, 23, 29]

# Mapping of end digit → 2-bit code
END_DIGIT_TO_BITS = {1: 0b00, 3: 0b01, 7: 0b10, 9: 0b11}
BITS_TO_END_DIGIT = {v: k for k, v in END_DIGIT_TO_BITS.items()}

# Pairing of the 8 residues into the four end-digit classes
# (each end digit owns exactly two positions per 30-block)
END_DIGIT_PAIRS = {
    1: [1, 11],
    7: [7, 17],
    3: [13, 23],
    9: [19, 29],
}


def end_digit(n: int) -> int:
    return n % 10


def to_binary_counter_plus_digit(p: int) -> Tuple[int, int, int]:
    """
    Represent prime p as 10*(base + counter) + digit.
    Returns (counter, digit, bits_of_digit).
    The 'base' is absorbed into the counter for relative stepping.
    """
    d = end_digit(p)
    if d not in END_DIGIT_TO_BITS:
        raise ValueError(f"{p} has forbidden end digit {d}")
    # Remove the last digit to get the higher part
    higher = p // 10
    bits = END_DIGIT_TO_BITS[d]
    return higher, d, bits


def block_index(n: int) -> int:
    """Which 30-block does n fall into?"""
    return n // 30


def residue_index(n: int) -> int:
    """Which of the 8 slots (0-7) does n occupy inside its 30-block?"""
    r = n % 30
    try:
        return RESIDUES_MOD30.index(r)
    except ValueError:
        return -1  # not a possible prime slot


def byte_for_block(start: int, primes_set: set) -> int:
    """
    Build the 8-bit byte for the block of 30 integers beginning at `start`
    (start must be a multiple of 30).
    Bit i is set if the residue RESIDUES_MOD30[i] + start is prime.
    """
    byte = 0
    for i, res in enumerate(RESIDUES_MOD30):
        candidate = start + res
        if candidate in primes_set:
            byte |= (1 << i)
    return byte


def generate_subset_bytes(limit: int) -> Dict[int, int]:
    """
    Generate the subset-byte map for all 30-blocks up to `limit`.
    Returns {block_index: byte_value}.
    """
    primes = set(sympy.primerange(5, limit + 30))
    max_block = limit // 30
    bytes_map = {}
    for b in range(max_block + 1):
        start = b * 30
        bytes_map[b] = byte_for_block(start, primes)
    return bytes_map


def rise_fall_bits(seq: List[int]) -> int:
    """
    Encode successive end-digit transitions as rise/fall bits.
    0 = rise, 1 = fall (so four consecutive 0s = four rises, which never occur).
    Rise/fall is defined on the cyclic order of admissible end digits:
        1 → 3 → 7 → 9 → 1 ...
    or simply by numerical comparison; the numerical version already
    reproduces the observed occupancy.
    """
    if len(seq) < 2:
        return 0
    code = 0
    for i in range(len(seq) - 1):
        # 0 = rise (next > current), 1 = fall
        if seq[i + 1] > seq[i]:
            # rise → bit stays 0
            pass
        else:
            # fall → set bit
            code |= (1 << i)
    return code


def analyze_subset_system(limit: int = 10**5):
    print("=== Binary Layer of the Subset System ===\n")
    print(f"Limit: {limit}")
    print(f"Admissible residues mod 30: {RESIDUES_MOD30}\n")

    # 1. Binary counter + 2-bit digit examples
    sample_primes = [102197, 102199, 102203, 102217, 102229, 102233, 102241, 102251, 102253]
    print("(A) Prime = binary counter + two-bit digit")
    print(f"{'prime':>8}  {'counter':>8}  {'digit':>5}  {'bits':>4}  formula")
    for p in sample_primes:
        if not sympy.isprime(p):
            continue
        higher, d, bits = to_binary_counter_plus_digit(p)
        print(f"{p:>8}  {higher:>8}  {d:>5}  {bits:02b}   10*({higher}) + {d}")

    # 2. Subset bytes
    print("\n(B) Subset bytes (one byte per 30-block)")
    bytes_map = generate_subset_bytes(limit)
    byte_counts = Counter(bytes_map.values())
    print(f"Total blocks examined: {len(bytes_map)}")
    print(f"Distinct byte values observed: {len(byte_counts)}")
    # Hottest bytes
    hottest = byte_counts.most_common(5)
    print("Hottest bytes (value, count, %):")
    total = len(bytes_map)
    for val, cnt in hottest:
        print(f"  0x{val:02X}  {cnt:6d}  {100*cnt/total:5.2f}%")

    # 3. Rise/fall skeleton on sequences of end digits
    primes = list(sympy.primerange(7, limit))
    end_digits = [end_digit(p) for p in primes]
    print(f"\n(C) Rise/fall skeleton on {len(end_digits)} end digits")
    # Sliding windows of length 8 (7 bits)
    window = 8
    rise_fall_counts = Counter()
    for i in range(len(end_digits) - window + 1):
        seq = end_digits[i:i + window]
        code = rise_fall_bits(seq)
        rise_fall_counts[code] += 1

    possible = 1 << (window - 1)  # 128
    occurred = len(rise_fall_counts)
    print(f"Windows of {window} digits → {window-1} bits: {occurred} of {possible} codes occurred")
    # Codes that never occur
    missing = [c for c in range(possible) if c not in rise_fall_counts]
    print(f"Missing codes: {len(missing)}")
    # Check how many missing contain four consecutive 0s (four rises)
    def has_four_zeros(c, bits=7):
        s = f"{c:0{bits}b}"
        return "0000" in s
    missing_with_0000 = sum(1 for c in missing if has_four_zeros(c))
    print(f"Of the missing codes, {missing_with_0000} contain '0000' (four rises)")

    # 4. Law at every length (small demonstration)
    print("\n(D) Occurrence law for short sequences")
    for length in [4, 5, 8]:
        bits = length - 1
        total_possible = 1 << bits
        counts = Counter()
        for i in range(len(end_digits) - length + 1):
            seq = end_digits[i:i + length]
            code = rise_fall_bits(seq)
            counts[code] += 1
        occurred = len(counts)
        print(f"  length {length:2d} ({bits} bits): {occurred:3d} of {total_possible} codes occurred")

    print("\n=== End of Subset Binary Layer Demo ===")


if __name__ == "__main__":
    analyze_subset_system(limit=100000)
