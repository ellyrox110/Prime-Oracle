# EJ Binary Subset Prime Oracle

A simple, explainable prime checker built on the **Binary Layer of the Subset System**  
(from Elizabeth Johnson’s Base Prime Model & Twin Prime Counterflow Model).

## What it does
- Instantly rejects numbers that violate the modular rules (end digit not in {1,3,7,9}, or not coprime to 30).
- Gives an exact yes/no via pure-Python trial division.
- Supplies a microscopic two-layer neural-net score trained on binary-subset features (rise/fall skeleton, twin flag, block density, gap residue, etc.).
- Explains every answer in the language of the model.

## Requirements
Python 3 + NumPy only.

## Usage
```bash
# Interactive chat
python prime_oracle.py

# Check one number
python prime_oracle.py 102197

# Quick demo
python prime_oracle.py --demo
```

## Example
```
Number: 102197
Binary form: 10×10219 + 7 (digit bits 10)
Mod-30 slot: 4 / 7
Exact verdict: PRIME
Micro-model score: 0.812
```

## Files
- `prime_oracle.py` — the complete, shareable oracle
- `prime_oracle_llm.py` — earlier developmental version
- `prime_oracle_perf_charts.png` — performance visualization

Feel free to use, share, and extend.  
Based on EJ’s Alternate Base Prime Number Pattern and the Twin Prime Counterflow Model.
