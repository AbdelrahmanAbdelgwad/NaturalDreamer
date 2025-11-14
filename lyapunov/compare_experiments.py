"""
Comparison script to run experiments with and without Lyapunov regularization
and visualize the differences in stability and performance.
"""

import os
import subprocess
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


def run_experiment(config_file, experiment_name):
    """Run training with specified config file."""
    print(f"Running experiment: {experiment_name}")
    subprocess.run(["python", "main_lyapunov.py", "--config", config_file])


def compare_metrics(baseline_csv, lyapunov_csv, output_path="comparison_plot.html"):
    """Compare metrics from baseline and Lyapunov experiments."""
    
    # Load data
    baseline_data = pd.read_csv(baseline_csv)
    lyapunov_data = pd.read_csv(lyapunov_csv)
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Episode Reward", 
            "Actor Loss",
            "Critic Loss",
            "Lyapunov Penalty",
            "Entropy",
            "Stability Metrics"
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )
    
    # Smoothing window
    window = 10
    
    # Plot Episode Reward
    fig.add_trace(
        go.Scatter(
            x=baseline_data["gradientSteps"],
            y=baseline_data["totalReward"].rolling(window=window, min_periods=1).mean(),
            name="Baseline",
            line=dict(color="blue", width=2)
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=lyapunov_data["gradientSteps"],
            y=lyapunov_data["totalReward"].rolling(window=window, min_periods=1).mean(),
            name="Lyapunov",
            line=dict(color="red", width=2)
        ),
        row=1, col=1
    )
    
    # Plot Actor Loss
    fig.add_trace(
        go.Scatter(
            x=baseline_data["gradientSteps"],
            y=baseline_data["actorLoss"].rolling(window=window, min_periods=1).mean(),
            name="Baseline Actor",
            line=dict(color="blue", width=2),
            showlegend=False
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=lyapunov_data["gradientSteps"],
            y=lyapunov_data["actorLoss"].rolling(window=window, min_periods=1).mean(),
            name="Lyapunov Actor",
            line=dict(color="red", width=2),
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Plot Critic Loss
    fig.add_trace(
        go.Scatter(
            x=baseline_data["gradientSteps"],
            y=baseline_data["criticLoss"].rolling(window=window, min_periods=1).mean(),
            name="Baseline Critic",
            line=dict(color="blue", width=2),
            showlegend=False
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=lyapunov_data["gradientSteps"],
            y=lyapunov_data["criticLoss"].rolling(window=window, min_periods=1).mean(),
            name="Lyapunov Critic",
            line=dict(color="red", width=2),
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Plot Lyapunov Penalty (only for Lyapunov version)
    if "lyapunovPenalty" in lyapunov_data.columns:
        fig.add_trace(
            go.Scatter(
                x=lyapunov_data["gradientSteps"],
                y=lyapunov_data["lyapunovPenalty"].rolling(window=window, min_periods=1).mean(),
                name="Lyapunov Penalty",
                line=dict(color="green", width=2)
            ),
            row=2, col=2
        )
        
    # Plot Entropy
    fig.add_trace(
        go.Scatter(
            x=baseline_data["gradientSteps"],
            y=baseline_data["entropies"].rolling(window=window, min_periods=1).mean(),
            name="Baseline Entropy",
            line=dict(color="blue", width=2),
            showlegend=False
        ),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=lyapunov_data["gradientSteps"],
            y=lyapunov_data["entropies"].rolling(window=window, min_periods=1).mean(),
            name="Lyapunov Entropy",
            line=dict(color="red", width=2),
            showlegend=False
        ),
        row=3, col=1
    )
    
    # Plot Lyapunov Mean (stability indicator)
    if "lyapunovMean" in lyapunov_data.columns:
        fig.add_trace(
            go.Scatter(
                x=lyapunov_data["gradientSteps"],
                y=lyapunov_data["lyapunovMean"].rolling(window=window, min_periods=1).mean(),
                name="Lyapunov Mean",
                line=dict(color="purple", width=2)
            ),
            row=3, col=2
        )
    
    # Update layout
    fig.update_layout(
        title="Baseline vs Lyapunov Regularization Comparison",
        height=900,
        width=1400,
        template="plotly_white",
        legend=dict(
            x=1.05,
            y=1,
            xanchor="left",
            yanchor="top"
        )
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Gradient Steps", row=3, col=1)
    fig.update_xaxes(title_text="Gradient Steps", row=3, col=2)
    
    fig.update_yaxes(title_text="Reward", row=1, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=2)
    fig.update_yaxes(title_text="Loss", row=2, col=1)
    fig.update_yaxes(title_text="Penalty", row=2, col=2)
    fig.update_yaxes(title_text="Entropy", row=3, col=1)
    fig.update_yaxes(title_text="Lyapunov Value", row=3, col=2)
    
    fig.write_html(output_path)
    print(f"Comparison plot saved to {output_path}")
    
    # Print statistics
    print("\n=== Performance Statistics ===")
    print(f"Baseline - Mean Reward (last 20%): {baseline_data['totalReward'].tail(len(baseline_data)//5).mean():.2f}")
    print(f"Lyapunov - Mean Reward (last 20%): {lyapunov_data['totalReward'].tail(len(lyapunov_data)//5).mean():.2f}")
    
    if "lyapunovPenalty" in lyapunov_data.columns:
        print(f"\n=== Stability Statistics ===")
        print(f"Final Lyapunov Penalty: {lyapunov_data['lyapunovPenalty'].iloc[-1]:.4f}")
        print(f"Mean Lyapunov Value: {lyapunov_data['lyapunovMean'].tail(100).mean():.4f}")


def create_ablation_configs(base_config="cartpole_lyapunov.yml"):
    """Create config files for ablation study with different lambda values."""
    import yaml
    
    with open(base_config, 'r') as f:
        config = yaml.safe_load(f)
    
    lambda_values = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
    
    for lambda_val in lambda_values:
        config['dreamer']['lyapunovLambda'] = lambda_val
        config['runName'] = f"CartpoleSwingup-Lambda{lambda_val}"
        
        filename = f"cartpole_lambda_{lambda_val}.yml"
        with open(filename, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"Created config: {filename}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare Dreamer experiments")
    parser.add_argument("--mode", choices=["run", "compare", "ablation"], default="compare",
                      help="Mode: run experiments, compare results, or run ablation study")
    parser.add_argument("--baseline_csv", default="metrics/cartpole-swingup_CartpoleSwingup-1.csv",
                      help="Path to baseline metrics CSV")
    parser.add_argument("--lyapunov_csv", default="metrics_lyapunov/cartpole-swingup_CartpoleSwingup-Lyapunov-1.csv",
                      help="Path to Lyapunov metrics CSV")
    
    args = parser.parse_args()
    
    if args.mode == "run":
        # Run both experiments
        run_experiment("cartpole.yml", "Baseline Dreamer")
        run_experiment("cartpole_lyapunov.yml", "Dreamer with Lyapunov")
        
    elif args.mode == "compare":
        # Compare results
        compare_metrics(args.baseline_csv, args.lyapunov_csv)
        
    elif args.mode == "ablation":
        # Create and run ablation study
        create_ablation_configs()
        print("Ablation configs created. Run each with:")
        print("python main_lyapunov.py --config cartpole_lambda_X.yml")
