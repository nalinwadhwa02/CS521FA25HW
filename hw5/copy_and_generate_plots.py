#!/usr/bin/env python3
import shutil
import json
import os
from pathlib import Path

# Set up matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Copy existing Figure 1 plots
source_dir = Path("refusal_direction/pipeline/runs/gemma-2b-it/select_direction")
dest_dir = Path(".")

for plot_file in ['actadd_scores.png', 'ablation_scores.png', 'kl_div_scores.png']:
    src = source_dir / plot_file
    dst = dest_dir / f"figure1_{plot_file}"
    if src.exists():
        shutil.copy(src, dst)
        print(f"Copied: {dst}")

# Generate Figure 3
model = "gemma-2b-it"
base_dir = Path("refusal_direction/pipeline/runs")

# Load jailbreak evaluation data
with open(base_dir / model / "completions/jailbreakbench_baseline_evaluations.json") as f:
    baseline = json.load(f)
with open(base_dir / model / "completions/jailbreakbench_ablation_evaluations.json") as f:
    ablation = json.load(f)
with open(base_dir / model / "completions/jailbreakbench_actadd_evaluations.json") as f:
    actadd = json.load(f)

# Create Figure 3
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

interventions = ['Baseline', 'Ablation', 'ActAdd\n(subtract direction)']
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

bars1 = ax.bar(x - width/2, substring_scores, width, label='Substring Matching', alpha=0.8, color='steelblue')
bars2 = ax.bar(x + width/2, llamaguard_scores, width, label='LlamaGuard2', alpha=0.8, color='coral')

ax.set_ylabel('Jailbreak Success Rate', fontsize=12)
ax.set_title(f'Figure 3: Jailbreak Success on Harmful Prompts\\n{model}', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(interventions)
ax.legend(fontsize=10)
ax.set_ylim([0, 1.0])
ax.grid(True, alpha=0.3, axis='y')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
               f'{height:.1%}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('figure3_jailbreak_results.png', dpi=150, bbox_inches='tight')
print("Generated: figure3_jailbreak_results.png")
plt.close()

# Print detailed summary
print(f"\\n{'='*60}")
print(f"Results for {model}")
print(f"{'='*60}\\n")

print("Jailbreak Success Rates on Harmful Prompts:")
print(f"  Baseline:")
print(f"    - Substring Matching: {baseline['substring_matching_success_rate']:.1%}")
print(f"    - LlamaGuard2: {baseline['llamaguard2_success_rate']:.1%}")
print(f"  Ablation (remove direction):")
print(f"    - Substring Matching: {ablation['substring_matching_success_rate']:.1%}")
print(f"    - LlamaGuard2: {ablation['llamaguard2_success_rate']:.1%}")
print(f"  ActAdd (subtract direction):")
print(f"    - Substring Matching: {actadd['substring_matching_success_rate']:.1%}")
print(f"    - LlamaGuard2: {actadd['llamaguard2_success_rate']:.1%}")

print(f"\\nKey Finding:")
improvement_substring = (actadd['substring_matching_success_rate'] - baseline['substring_matching_success_rate']) / baseline['substring_matching_success_rate'] * 100
improvement_llama = (actadd['llamaguard2_success_rate'] - baseline['llamaguard2_success_rate']) / (baseline['llamaguard2_success_rate'] + 0.01) * 100

print(f"  Activation addition (subtracting the refusal direction) increases")
print(f"  jailbreak success by {improvement_substring:.0f}% (substring) / {improvement_llama:.0f}% (LlamaGuard2)")
print(f"  compared to baseline, demonstrating that a single direction mediates refusal.")

print(f"\\n{'='*60}")
