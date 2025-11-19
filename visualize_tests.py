import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path


def plot_comparison(df, output_dir="test_plots"):
    """Create comparison plots for all metrics."""
    Path(output_dir).mkdir(exist_ok=True)
    
    standard_df = df[df['test_type'] == 'standard']
    
    # 1. Return comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    runs = standard_df['run_name'].values
    returns = standard_df['mean_return'].values
    stds = standard_df['std_return'].values
    
    x = np.arange(len(runs))
    colors = ['#1f77b4' if 'Baseline' in r else '#ff7f0e' for r in runs]
    ax.bar(x, returns, yerr=stds, capsize=5, alpha=0.7, color=colors)
    ax.set_xlabel('Run')
    ax.set_ylabel('Mean Return')
    ax.set_title('Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(runs, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/return_comparison.png", dpi=150)
    plt.close()
    
    # 2. Failure rate comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    failure_rates = standard_df['failure_rate'].values * 100
    ax.bar(x, failure_rates, alpha=0.7, color=colors)
    ax.set_xlabel('Run')
    ax.set_ylabel('Failure Rate (%)')
    ax.set_title('Failure Rate Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(runs, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/failure_rate_comparison.png", dpi=150)
    plt.close()
    
    # 3. Action norm comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    action_norms = standard_df['mean_action_norm'].values
    action_stds = standard_df['std_action_norm'].values
    ax.bar(x, action_norms, yerr=action_stds, capsize=5, alpha=0.7, color=colors)
    ax.set_xlabel('Run')
    ax.set_ylabel('Mean Action Norm')
    ax.set_title('Action Norm Comparison (Behavioral Smoothness)')
    ax.set_xticks(x)
    ax.set_xticklabels(runs, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/action_norm_comparison.png", dpi=150)
    plt.close()
    
    # 4. Robustness plot (if available)
    robustness_df = df[df['test_type'] == 'robustness']
    if not robustness_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for run_name in robustness_df['run_name'].unique():
            run_data = robustness_df[robustness_df['run_name'] == run_name]
            noise_levels = run_data['noise_std'].values
            returns = run_data['mean_return'].values
            
            linestyle = '--' if 'Baseline' in run_name else '-'
            ax.plot(noise_levels, returns, marker='o', label=run_name, linestyle=linestyle)
        
        ax.set_xlabel('Noise Standard Deviation (σ)')
        ax.set_ylabel('Mean Return')
        ax.set_title('Robustness Under Gaussian Noise')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/robustness_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Plots saved to {output_dir}/")


def print_summary_table(df):
    """Print formatted summary table."""
    standard_df = df[df['test_type'] == 'standard']
    
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    print(f"{'Run':<35} {'Return':<18} {'Failure %':<12} {'Action Norm':<12}")
    print("-"*80)
    
    for _, row in standard_df.iterrows():
        run_name = row['run_name'][:33]
        return_str = f"{row['mean_return']:.1f} ± {row['std_return']:.1f}"
        failure_str = f"{row['failure_rate']*100:.1f}%"
        action_str = f"{row['mean_action_norm']:.3f}"
        print(f"{run_name:<35} {return_str:<18} {failure_str:<12} {action_str:<12}")
    
    print("="*80)
    
    # Robustness degradation
    robustness_df = df[df['test_type'] == 'robustness']
    if not robustness_df.empty:
        print("\nROBUSTNESS DEGRADATION (σ=0.0 → σ=0.1)")
        print("="*80)
        
        for run_name in robustness_df['run_name'].unique():
            run_data = robustness_df[robustness_df['run_name'] == run_name]
            clean_data = run_data[run_data['noise_std'] == 0.0]
            noisy_data = run_data[run_data['noise_std'] == 0.1]
            
            if not clean_data.empty and not noisy_data.empty:
                clean = clean_data['mean_return'].values[0]
                noisy = noisy_data['mean_return'].values[0]
                degradation = ((clean - noisy) / clean) * 100 if clean > 0 else 0
                
                print(f"{run_name:<35} {clean:.1f} → {noisy:.1f} ({degradation:.1f}% drop)")


def main(args):
    df = pd.read_csv(args.results_csv)
    
    print_summary_table(df)
    
    if args.plot:
        plot_comparison(df, args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize checkpoint test results")
    parser.add_argument("--results-csv", type=str, default="test_results.csv", help="CSV with results")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument("--output-dir", type=str, default="test_plots", help="Output directory for plots")
    
    args = parser.parse_args()
    main(args)
