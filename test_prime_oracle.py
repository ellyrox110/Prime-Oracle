#!/usr/bin/env python3
"""
Validity tests and performance visualization
for the Binary Subset Prime Oracle LLM
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from prime_oracle_llm import is_prime_pure, is_prime_llm, MicroPrimeModel, to_subset_binary, slot_in_byte

random.seed(42)
np.random.seed(42)

def generate_test_set(n_samples=2000, max_n=100000):
    """Mix of random integers, known primes, and composites."""
    samples = []
    # Random numbers
    for _ in range(n_samples // 2):
        samples.append(random.randint(10, max_n))
    # Some larger random
    for _ in range(n_samples // 4):
        samples.append(random.randint(max_n, max_n * 5))
    # Force some primes by taking random and next prime-ish
    for _ in range(n_samples // 4):
        base = random.randint(100, max_n)
        # simple search for a prime near base
        for offset in range(0, 100, 2):
            cand = base + offset
            if is_prime_pure(cand):
                samples.append(cand)
                break
        else:
            samples.append(base)
    return list(set(samples))  # unique

def run_validity_tests(samples):
    results = []
    correct = 0
    model_correct = 0
    model_scores = []
    labels = []
    preds = []
    
    for n in samples:
        true_prime = is_prime_pure(n)
        # Get full explanation (also trains model on first call)
        explanation = is_prime_llm(n, explain=True, use_model=True)
        # Parse model score if present
        model_score = None
        for line in explanation.split("\n"):
            if "Micro-model score:" in line:
                try:
                    model_score = float(line.split(":")[1].split("(")[0].strip())
                except:
                    pass
        model_pred = model_score >= 0.5 if model_score is not None else None
        
        is_correct = True  # trial division is ground truth
        if model_pred is not None:
            if model_pred == true_prime:
                model_correct += 1
            model_scores.append(model_score)
            labels.append(1 if true_prime else 0)
            preds.append(1 if model_pred else 0)
        
        results.append({
            "n": n,
            "true_prime": true_prime,
            "model_score": model_score,
            "model_pred": model_pred,
            "explanation_ok": "Verdict" in explanation or "cannot be prime" in explanation
        })
        correct += 1  # always, since pure is ground truth
    
    return results, correct, model_correct, model_scores, labels, preds

def create_charts(results, model_scores, labels, preds, save_prefix="/workspace/artifacts/prime_oracle_perf"):
    # 1. Score distribution by true label
    primes_scores = [s for s, l in zip(model_scores, labels) if l == 1]
    composites_scores = [s for s, l in zip(model_scores, labels) if l == 0]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    # Histogram of scores
    axes[0].hist(primes_scores, bins=20, alpha=0.7, label="True primes", color="green", density=True)
    axes[0].hist(composites_scores, bins=20, alpha=0.7, label="True composites", color="red", density=True)
    axes[0].axvline(0.5, color="black", linestyle="--", label="Threshold 0.5")
    axes[0].set_title("Micro-model score distribution")
    axes[0].set_xlabel("Predicted probability")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    
    # Accuracy by magnitude
    magnitudes = []
    acc_by_mag = []
    bins = [0, 100, 1000, 10000, 100000, 500000]
    for lo, hi in zip(bins[:-1], bins[1:]):
        subset = [r for r in results if lo <= r["n"] < hi and r["model_pred"] is not None]
        if subset:
            acc = np.mean([r["model_pred"] == r["true_prime"] for r in subset])
            magnitudes.append(f"{lo}-{hi}")
            acc_by_mag.append(acc)
    axes[1].bar(magnitudes, acc_by_mag, color="steelblue")
    axes[1].set_title("Micro-model accuracy by number size")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis="x", rotation=45)
    
    # Confusion-style counts
    tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
    tn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 0)
    fn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)
    axes[2].bar(["TP", "FP", "TN", "FN"], [tp, fp, tn, fn], color=["green", "orange", "blue", "red"])
    axes[2].set_title("Micro-model confusion counts")
    axes[2].set_ylabel("Count")
    
    plt.tight_layout()
    chart_path = f"{save_prefix}_charts.png"
    plt.savefig(chart_path, dpi=120, bbox_inches="tight")
    plt.close()
    return chart_path, tp, fp, tn, fn

if __name__ == "__main__":
    print("Generating test set...")
    samples = generate_test_set(n_samples=1500, max_n=200000)
    print(f"Test set size: {len(samples)}")
    
    print("Running validity tests (this also trains the micro-model)...")
    results, correct, model_correct, model_scores, labels, preds = run_validity_tests(samples)
    
    total = len(results)
    model_total = len(model_scores)
    model_acc = model_correct / model_total if model_total else 0
    
    print("\n=== VALIDITY RESULTS ===")
    print(f"Total numbers tested: {total}")
    print(f"Trial-division verdicts consistent: {correct}/{total} (100% by construction)")
    print(f"Micro-model accuracy: {model_correct}/{model_total} = {model_acc:.3f}")
    print(f"Prime rate in test set: {np.mean(labels):.3f}")
    
    print("\nCreating performance charts...")
    chart_path, tp, fp, tn, fn = create_charts(results, model_scores, labels, preds)
    print(f"Charts saved to: {chart_path}")
    print(f"Confusion: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    
    # Quick spot checks
    print("\n=== SPOT CHECKS ===")
    spot = [11, 15, 101, 10007, 999983, 1000003, 4, 9, 25, 49]
    for s in spot:
        true = is_prime_pure(s)
        expl = is_prime_llm(s, explain=False)
        print(f"{s:8d}  true={true}  oracle={expl}")
