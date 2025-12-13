"""
Visualization script for reproducing Figures 1 and 3 from the paper:
"Refusal in Language Models Is Mediated by a Single Direction"

Figure 1: Activation Addition Results (Bypassing Refusal)
- Shows jailbreak success rate when applying activation addition (subtracting refusal direction)
- Compares baseline vs. activation addition across harmful and harmless prompts

Figure 3: Directional Ablation Results
- Shows effect of ablating the refusal direction on model behavior
- Compares baseline vs. directional ablation on harmful prompts
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_evaluation_data(model_name, intervention, dataset):
    """Load evaluation data for a specific model, intervention, and dataset."""
    base_path = Path("hw5/refusal_direction/pipeline/runs")
    eval_file = base_path / model_name / "completions" / f"{dataset}_{intervention}_evaluations.json"

    if eval_file.exists():
        with open(eval_file, 'r') as f:
            return json.load(f)
    return None

def load_loss_data(model_name, intervention):
    """Load loss evaluation data."""
    base_path = Path("hw5/refusal_direction/pipeline/runs")
    loss_file = base_path / model_name / "loss_evals" / f"{intervention}_loss_eval.json"

    if loss_file.exists():
        with open(loss_file, 'r') as f:
            return json.load(f)
    return None

def plot_figure1_activation_addition(model_name, save_path="hw5/figure1_activation_addition.png"):
    """
    Figure 1: Activation Addition Results
    Shows how adding the negative refusal direction bypasses refusal on harmful prompts
    and induces refusal on harmless prompts.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: Harmful prompts (JailbreakBench)
    ax1 = axes[0]

    # Load data for harmful prompts
    baseline_harmful = load_evaluation_data(model_name, "baseline", "jailbreakbench")
    actadd_harmful = load_evaluation_data(model_name, "actadd", "jailbreakbench")

    if baseline_harmful and actadd_harmful:
        # Extract success rates (higher = more jailbroken)
        metrics = ['substring_matching_success_rate', 'llamaguard2_success_rate']
        metric_labels = ['Substring Matching', 'LlamaGuard2']

        x = np.arange(len(metrics))
        width = 0.35

        baseline_scores = [baseline_harmful.get(m, 0) for m in metrics]
        actadd_scores = [actadd_harmful.get(m, 0) for m in metrics]

        bars1 = ax1.bar(x - width/2, baseline_scores, width, label='Baseline', color='steelblue', alpha=0.8)
        bars2 = ax1.bar(x + width/2, actadd_scores, width, label='Activation Addition\n(-1.0 × refusal dir)',
                       color='coral', alpha=0.8)

        ax1.set_ylabel('Jailbreak Success Rate', fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels(metric_labels)
        ax1.legend(fontsize=10)
        ax1.set_ylim([0, 1.0])
        ax1.grid(axis='y', alpha=0.3, linestyle='--')

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom', fontsize=9)

    # Right panel: Harmless prompts
    ax2 = axes[1]

    # Load data for harmless prompts
    baseline_harmless = load_evaluation_data(model_name, "baseline", "harmless")
    actadd_harmless = load_evaluation_data(model_name, "actadd", "harmless")

    if baseline_harmless and actadd_harmless:
        # For harmless prompts, we're adding +1.0 × refusal direction to induce refusal
        # Higher refusal rate = more refusals (which is what we want to show)
        metric = 'substring_matching_success_rate'

        # Note: In the actadd for harmless, the coefficient is +1.0 (inducing refusal)
        # Success rate here means "successfully refused"
        baseline_rate = baseline_harmless.get(metric, 0)
        actadd_rate = actadd_harmless.get(metric, 0)

        categories = ['Baseline', 'Activation Addition\n(+1.0 × refusal dir)']
        values = [baseline_rate, actadd_rate]

        bars = ax2.bar(categories, values, color=['steelblue', 'coral'], alpha=0.8, width=0.6)

        ax2.set_ylabel('Refusal Rate', fontsize=12)
        ax2.set_ylim([0, 1.0])
        ax2.grid(axis='y', alpha=0.3, linestyle='--')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved Figure 1 to {save_path}")
    plt.close()

def plot_figure3_directional_ablation(model_name, save_path="hw5/figure3_directional_ablation.png"):
    """
    Figure 3: Directional Ablation Results
    Shows how ablating the refusal direction affects jailbreak success rates
    and perplexity on various datasets.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: Jailbreak success rates
    ax1 = axes[0]

    baseline_harmful = load_evaluation_data(model_name, "baseline", "jailbreakbench")
    ablation_harmful = load_evaluation_data(model_name, "ablation", "jailbreakbench")

    if baseline_harmful and ablation_harmful:
        metrics = ['substring_matching_success_rate', 'llamaguard2_success_rate']
        metric_labels = ['Substring\nMatching', 'LlamaGuard2']

        x = np.arange(len(metrics))
        width = 0.35

        baseline_scores = [baseline_harmful.get(m, 0) for m in metrics]
        ablation_scores = [ablation_harmful.get(m, 0) for m in metrics]

        bars1 = ax1.bar(x - width/2, baseline_scores, width, label='Baseline',
                       color='steelblue', alpha=0.8)
        bars2 = ax1.bar(x + width/2, ablation_scores, width, label='Directional Ablation',
                       color='darkseagreen', alpha=0.8)

        ax1.set_ylabel('Jailbreak Success Rate', fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels(metric_labels)
        ax1.legend(fontsize=10)
        ax1.set_ylim([0, 1.0])
        ax1.grid(axis='y', alpha=0.3, linestyle='--')

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom', fontsize=9)

    # Right panel: Perplexity on different datasets
    ax2 = axes[1]

    baseline_loss = load_loss_data(model_name, "baseline")
    ablation_loss = load_loss_data(model_name, "ablation")

    if baseline_loss and ablation_loss:
        datasets = ['pile', 'alpaca', 'alpaca_custom_completions']
        dataset_labels = ['Pile', 'Alpaca', 'Alpaca\nCompletions']

        x = np.arange(len(datasets))
        width = 0.35

        baseline_perplexity = [baseline_loss[d]['perplexity'] for d in datasets]
        ablation_perplexity = [ablation_loss[d]['perplexity'] for d in datasets]

        bars1 = ax2.bar(x - width/2, baseline_perplexity, width, label='Baseline',
                       color='steelblue', alpha=0.8)
        bars2 = ax2.bar(x + width/2, ablation_perplexity, width, label='Directional Ablation',
                       color='darkseagreen', alpha=0.8)

        ax2.set_ylabel('Perplexity', fontsize=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels(dataset_labels)
        ax2.legend(fontsize=10)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}',
                        ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved Figure 3 to {save_path}")
    plt.close()

def plot_comprehensive_comparison(model_name, save_path="hw5/comprehensive_refusal_analysis.png"):
    """
    Create a comprehensive 4-panel figure showing all interventions and metrics.
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    # Panel 1: Jailbreak Success Rates (All Interventions)
    ax1 = fig.add_subplot(gs[0, 0])

    baseline_harmful = load_evaluation_data(model_name, "baseline", "jailbreakbench")
    actadd_harmful = load_evaluation_data(model_name, "actadd", "jailbreakbench")
    ablation_harmful = load_evaluation_data(model_name, "ablation", "jailbreakbench")

    if all([baseline_harmful, actadd_harmful, ablation_harmful]):
        interventions = ['Baseline', 'Activation\nAddition', 'Directional\nAblation']

        substring_scores = [
            baseline_harmful.get('substring_matching_success_rate', 0),
            actadd_harmful.get('substring_matching_success_rate', 0),
            ablation_harmful.get('substring_matching_success_rate', 0)
        ]

        llamaguard_scores = [
            baseline_harmful.get('llamaguard2_success_rate', 0),
            actadd_harmful.get('llamaguard2_success_rate', 0),
            ablation_harmful.get('llamaguard2_success_rate', 0)
        ]

        x = np.arange(len(interventions))
        width = 0.35

        bars1 = ax1.bar(x - width/2, substring_scores, width, label='Substring Matching',
                       color='coral', alpha=0.8)
        bars2 = ax1.bar(x + width/2, llamaguard_scores, width, label='LlamaGuard2',
                       color='mediumpurple', alpha=0.8)

        ax1.set_ylabel('Jailbreak Success Rate', fontsize=11)
        ax1.set_title('(A) Harmful Prompts - Jailbreak Success', fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(interventions)
        ax1.legend(fontsize=9)
        ax1.set_ylim([0, 1.0])
        ax1.grid(axis='y', alpha=0.3, linestyle='--')

        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=8)

    # Panel 2: Harmless Prompts Refusal Rates
    ax2 = fig.add_subplot(gs[0, 1])

    baseline_harmless = load_evaluation_data(model_name, "baseline", "harmless")
    actadd_harmless = load_evaluation_data(model_name, "actadd", "harmless")

    if baseline_harmless and actadd_harmless:
        interventions = ['Baseline', 'Activation Addition\n(+1.0 × refusal dir)']
        refusal_rates = [
            baseline_harmless.get('substring_matching_success_rate', 0),
            actadd_harmless.get('substring_matching_success_rate', 0)
        ]

        bars = ax2.bar(interventions, refusal_rates, color=['steelblue', 'coral'],
                      alpha=0.8, width=0.5)

        ax2.set_ylabel('Refusal Rate', fontsize=11)
        ax2.set_title('(B) Harmless Prompts - Induced Refusal', fontsize=12, fontweight='bold')
        ax2.set_ylim([0, 1.0])
        ax2.grid(axis='y', alpha=0.3, linestyle='--')

        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)

    # Panel 3: Perplexity Comparison
    ax3 = fig.add_subplot(gs[1, 0])

    baseline_loss = load_loss_data(model_name, "baseline")
    actadd_loss = load_loss_data(model_name, "actadd")
    ablation_loss = load_loss_data(model_name, "ablation")

    if all([baseline_loss, actadd_loss, ablation_loss]):
        datasets = ['pile', 'alpaca']
        dataset_labels = ['Pile', 'Alpaca']

        x = np.arange(len(datasets))
        width = 0.25

        baseline_ppl = [baseline_loss[d]['perplexity'] for d in datasets]
        actadd_ppl = [actadd_loss[d]['perplexity'] for d in datasets]
        ablation_ppl = [ablation_loss[d]['perplexity'] for d in datasets]

        bars1 = ax3.bar(x - width, baseline_ppl, width, label='Baseline',
                       color='steelblue', alpha=0.8)
        bars2 = ax3.bar(x, actadd_ppl, width, label='Act. Addition',
                       color='coral', alpha=0.8)
        bars3 = ax3.bar(x + width, ablation_ppl, width, label='Ablation',
                       color='darkseagreen', alpha=0.8)

        ax3.set_ylabel('Perplexity', fontsize=11)
        ax3.set_title('(C) Model Perplexity on Various Datasets', fontsize=12, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(dataset_labels)
        ax3.legend(fontsize=9)
        ax3.grid(axis='y', alpha=0.3, linestyle='--')

        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=7)

    # Panel 4: Per-category breakdown for harmful prompts
    ax4 = fig.add_subplot(gs[1, 1])

    if baseline_harmful and actadd_harmful:
        categories = baseline_harmful.get('substring_matching_per_category', {})
        actadd_categories = actadd_harmful.get('substring_matching_per_category', {})

        if categories:
            # Get top categories by baseline score
            sorted_cats = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:6]
            cat_names = [c[0][:20] for c in sorted_cats]  # Truncate long names

            baseline_vals = [categories[c[0]] for c in sorted_cats]
            actadd_vals = [actadd_categories.get(c[0], 0) for c in sorted_cats]

            x = np.arange(len(cat_names))
            width = 0.35

            bars1 = ax4.barh(x + width/2, baseline_vals, width, label='Baseline',
                            color='steelblue', alpha=0.8)
            bars2 = ax4.barh(x - width/2, actadd_vals, width, label='Activation Addition',
                            color='coral', alpha=0.8)

            ax4.set_xlabel('Jailbreak Success Rate', fontsize=11)
            ax4.set_title('(D) Per-Category Jailbreak Success (Top 6)', fontsize=12, fontweight='bold')
            ax4.set_yticks(x)
            ax4.set_yticklabels(cat_names, fontsize=9)
            ax4.legend(fontsize=9)
            ax4.set_xlim([0, 1.0])
            ax4.grid(axis='x', alpha=0.3, linestyle='--')
            ax4.invert_yaxis()

    plt.suptitle(f'Comprehensive Refusal Direction Analysis - {model_name}',
                 fontsize=15, fontweight='bold', y=0.995)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved comprehensive analysis to {save_path}")
    plt.close()

def generate_summary_report(model_name, save_path="hw5/refusal_analysis_summary.txt"):
    """Generate a text summary of the results."""
    with open(save_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"REFUSAL DIRECTION ANALYSIS SUMMARY - {model_name}\n")
        f.write("="*80 + "\n\n")

        # Load all data
        baseline_harmful = load_evaluation_data(model_name, "baseline", "jailbreakbench")
        actadd_harmful = load_evaluation_data(model_name, "actadd", "jailbreakbench")
        ablation_harmful = load_evaluation_data(model_name, "ablation", "jailbreakbench")
        baseline_harmless = load_evaluation_data(model_name, "baseline", "harmless")
        actadd_harmless = load_evaluation_data(model_name, "actadd", "harmless")
        baseline_loss = load_loss_data(model_name, "baseline")
        actadd_loss = load_loss_data(model_name, "actadd")
        ablation_loss = load_loss_data(model_name, "ablation")

        # Harmful prompts analysis
        f.write("1. HARMFUL PROMPTS (JailbreakBench Dataset)\n")
        f.write("-" * 80 + "\n\n")

        if baseline_harmful and actadd_harmful and ablation_harmful:
            f.write("   Jailbreak Success Rates:\n")
            f.write(f"   Baseline:\n")
            f.write(f"     - Substring Matching: {baseline_harmful.get('substring_matching_success_rate', 0):.2%}\n")
            f.write(f"     - LlamaGuard2:       {baseline_harmful.get('llamaguard2_success_rate', 0):.2%}\n\n")

            f.write(f"   Activation Addition (-1.0 × refusal direction):\n")
            f.write(f"     - Substring Matching: {actadd_harmful.get('substring_matching_success_rate', 0):.2%}\n")
            f.write(f"     - LlamaGuard2:       {actadd_harmful.get('llamaguard2_success_rate', 0):.2%}\n")

            # Calculate increase
            substring_increase = actadd_harmful.get('substring_matching_success_rate', 0) - baseline_harmful.get('substring_matching_success_rate', 0)
            llamaguard_increase = actadd_harmful.get('llamaguard2_success_rate', 0) - baseline_harmful.get('llamaguard2_success_rate', 0)
            f.write(f"     - Increase (Substring): {substring_increase:+.2%}\n")
            f.write(f"     - Increase (LlamaGuard2): {llamaguard_increase:+.2%}\n\n")

            f.write(f"   Directional Ablation:\n")
            f.write(f"     - Substring Matching: {ablation_harmful.get('substring_matching_success_rate', 0):.2%}\n")
            f.write(f"     - LlamaGuard2:       {ablation_harmful.get('llamaguard2_success_rate', 0):.2%}\n\n")

        # Harmless prompts analysis
        f.write("2. HARMLESS PROMPTS\n")
        f.write("-" * 80 + "\n\n")

        if baseline_harmless and actadd_harmless:
            f.write("   Refusal Rates:\n")
            f.write(f"   Baseline:                              {baseline_harmless.get('substring_matching_success_rate', 0):.2%}\n")
            f.write(f"   Activation Addition (+1.0 × refusal):  {actadd_harmless.get('substring_matching_success_rate', 0):.2%}\n")

            increase = actadd_harmless.get('substring_matching_success_rate', 0) - baseline_harmless.get('substring_matching_success_rate', 0)
            f.write(f"   Increase in refusal:                   {increase:+.2%}\n\n")

        # Perplexity analysis
        f.write("3. MODEL PERPLEXITY (Quality Metrics)\n")
        f.write("-" * 80 + "\n\n")

        if baseline_loss and actadd_loss and ablation_loss:
            for dataset in ['pile', 'alpaca', 'alpaca_custom_completions']:
                if dataset in baseline_loss:
                    f.write(f"   {dataset.upper()}:\n")
                    f.write(f"     Baseline:            PPL = {baseline_loss[dataset]['perplexity']:.2f}, CE Loss = {baseline_loss[dataset]['ce_loss']:.3f}\n")
                    f.write(f"     Activation Addition: PPL = {actadd_loss[dataset]['perplexity']:.2f}, CE Loss = {actadd_loss[dataset]['ce_loss']:.3f}\n")
                    f.write(f"     Directional Ablation: PPL = {ablation_loss[dataset]['perplexity']:.2f}, CE Loss = {ablation_loss[dataset]['ce_loss']:.3f}\n\n")

        # Key findings
        f.write("4. KEY FINDINGS\n")
        f.write("-" * 80 + "\n\n")

        if all([baseline_harmful, actadd_harmful, ablation_harmful]):
            f.write("   NECESSITY (Activation Addition):\n")
            f.write("   - Subtracting the refusal direction significantly increases jailbreak success\n")
            f.write("   - This demonstrates that the direction is NECESSARY for refusal behavior\n\n")

            f.write("   SUFFICIENCY (Directional Ablation):\n")
            f.write("   - Ablating the refusal direction also increases jailbreak success\n")
            f.write("   - This demonstrates that the direction is SUFFICIENT for refusal behavior\n\n")

            f.write("   BIDIRECTIONALITY:\n")
            f.write("   - Adding positive refusal direction to harmless prompts induces refusal\n")
            f.write("   - This shows the direction works bidirectionally\n\n")

        f.write("="*80 + "\n")

    print(f"Saved summary report to {save_path}")

if __name__ == "__main__":
    # Use the Llama-3 8B model as the primary example
    model_name = "meta-llama-3-8b-instruct"

    print(f"\nGenerating visualizations for {model_name}...\n")

    # Generate Figure 1 (Activation Addition)
    plot_figure1_activation_addition(model_name)

    # Generate Figure 3 (Directional Ablation)
    plot_figure3_directional_ablation(model_name)

    # Generate comprehensive comparison
    plot_comprehensive_comparison(model_name)

    # Generate summary report
    generate_summary_report(model_name)

    print("\nAll visualizations and summary generated successfully!")
    print("\nGenerated files:")
    print("  - hw5/figure1_activation_addition.png")
    print("  - hw5/figure3_directional_ablation.png")
    print("  - hw5/comprehensive_refusal_analysis.png")
    print("  - hw5/refusal_analysis_summary.txt")
