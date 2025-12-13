#!/usr/bin/env python3
"""Generate figures for refusal direction analysis"""

import json
import sys
import os

# Add matplotlib backend
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

def main():
    model = "gemma-2b-it"
    base_dir = "refusal_direction/pipeline/runs"

    # Load jailbreak evaluation data
    with open(f"{base_dir}/{model}/completions/jailbreakbench_baseline_evaluations.json") as f:
        baseline = json.load(f)
    with open(f"{base_dir}/{model}/completions/jailbreakbench_ablation_evaluations.json") as f:
        ablation = json.load(f)
    with open(f"{base_dir}/{model}/completions/jailbreakbench_actadd_evaluations.json") as f:
        actadd = json.load(f)

    # Create Figure 3
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    interventions = ['Baseline', 'Ablation', 'ActAdd']
    substring_scores = [
        baseline['substring_matching_success_rate'],
        ablation['substring_matching_success_rate'],
        actadd['substring_matching_success_rate']
    ]
    llamaguard_scores = [
        baseline['llamaguard2_success_rate'],
        ablation['llamaguard2_success_rate'],
        actadd['llamaguard2_success_rate']
    ]

    x = np.arange(len(interventions))
    width = 0.35

    bars1 = ax.bar(x - width/2, substring_scores, width, label='Substring Matching')
    bars2 = ax.bar(x + width/2, llamaguard_scores, width, label='LlamaGuard2')

    ax.set_ylabel('Jailbreak Success Rate')
    ax.set_title(f'Jailbreak Success on Harmful Prompts - {model}')
    ax.set_xticks(x)
    ax.set_xticklabels(interventions)
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('figure3_jailbreak_results.png', dpi=150)
    print("Generated: figure3_jailbreak_results.png")

    # Print summary
    print(f"\nResults for {model}:")
    print(f"Baseline - Substring: {baseline['substring_matching_success_rate']:.1%}, LlamaGuard: {baseline['llamaguard2_success_rate']:.1%}")
    print(f"Ablation - Substring: {ablation['substring_matching_success_rate']:.1%}, LlamaGuard: {ablation['llamaguard2_success_rate']:.1%}")
    print(f"ActAdd - Substring: {actadd['substring_matching_success_rate']:.1%}, LlamaGuard: {actadd['llamaguard2_success_rate']:.1%}")

if __name__ == "__main__":
    main()
