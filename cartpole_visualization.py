import os
import argparse
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_EVAL_CSV = "eval_results/eval_cartpole.csv"
# This is what main.py would have produced for your cartpole run:
DEFAULT_METRICS_CSV = os.path.join(
    "metrics", "cartpole-swingup_CartpoleSwingup-1.csv"
)


# ---------- Helper utils ----------

def ensure_dir(path: str) -> None:
    """Create parent directory if it doesn't exist."""
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def load_eval_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "noise_sigma",
        "return",
        "length",
        "failed",
        "action_norm_mean",
        "action_norm_std",
        "agent_type",
        "seed",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"eval CSV is missing columns: {missing}")
    return df


def safe_load_training_csv(path: Optional[str]) -> Optional[pd.DataFrame]:
    if path is None or not os.path.exists(path):
        print(f"[Train] metrics CSV not found at {path}. "
              f"Training-based plots will be skipped.")
        return None

    df = pd.read_csv(path)
    return df


# ---------- Eval summaries ----------

def summarize_by_noise(df_eval: pd.DataFrame) -> pd.DataFrame:
    """Aggregate eval metrics per noise_sigma."""
    grouped = df_eval.groupby("noise_sigma")
    summary = grouped.agg(
        mean_return=("return", "mean"),
        std_return=("return", "std"),
        failure_rate=("failed", "mean"),
        mean_action_norm=("action_norm_mean", "mean"),
        std_action_norm=("action_norm_mean", "std"),
        mean_action_norm_std=("action_norm_std", "mean"),
    ).reset_index()

    # failure_rate currently in [0,1]; convert to %
    summary["failure_rate"] *= 100.0
    return summary


def compute_robustness_slopes(df_eval: pd.DataFrame) -> pd.DataFrame:
    """
    For each (agent_type, seed) compute the slope of Return vs noise_sigma.
    More negative slope => more fragile to noise.
    """
    rows = []
    for (agent, seed), g in df_eval.groupby(["agent_type", "seed"]):
        x = g["noise_sigma"].values
        y = g["return"].values
        if len(np.unique(x)) < 2:
            continue
        # Fit simple linear regression y = a * x + b
        a, b = np.polyfit(x, y, 1)
        rows.append(
            {
                "agent_type": agent,
                "seed": seed,
                "slope": a,
                "intercept": b,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["agent_type", "seed", "slope", "intercept"])
    return pd.DataFrame(rows)


# ---------- Plotting functions (EVAL) ----------

def plot_return_vs_noise(summary: pd.DataFrame, out_path: str) -> None:
    ensure_dir(out_path)
    plt.figure(figsize=(6, 4))
    x = summary["noise_sigma"].values
    y = summary["mean_return"].values
    yerr = summary["std_return"].values

    plt.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="o-",
        capsize=4,
    )
    plt.xlabel("Noise standard deviation σ")
    plt.ylabel("Return (episode reward)")
    plt.title("Task performance vs. injected dynamics noise")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[Plot] Saved return vs noise -> {out_path}")


def plot_failure_rate_vs_noise(summary: pd.DataFrame, out_path: str) -> None:
    ensure_dir(out_path)
    plt.figure(figsize=(6, 4))
    x = summary["noise_sigma"].values
    y = summary["failure_rate"].values

    plt.plot(x, y, "o-", linewidth=2)
    plt.xlabel("Noise standard deviation σ")
    plt.ylabel("Failure rate (%)")
    plt.title("Stability / Safety: failure rate vs. noise")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[Plot] Saved failure rate vs noise -> {out_path}")


def plot_action_norms_vs_noise(summary: pd.DataFrame, out_path: str) -> None:
    """
    One figure: mean action norm and mean action_norm_std across noise.
    """
    ensure_dir(out_path)
    plt.figure(figsize=(6, 4))

    x = summary["noise_sigma"].values
    mean_norm = summary["mean_action_norm"].values
    std_over_episodes = summary["std_action_norm"].values
    mean_inner_std = summary["mean_action_norm_std"].values

    plt.plot(x, mean_norm, "o-", label="Mean action norm")
    plt.fill_between(
        x,
        mean_norm - std_over_episodes,
        mean_norm + std_over_episodes,
        alpha=0.2,
        label="±1 std over episodes",
    )

    # Second line for typical within-episode variation
    plt.plot(
        x,
        mean_inner_std,
        "s--",
        label="Mean within-episode action std",
    )

    plt.xlabel("Noise standard deviation σ")
    plt.ylabel("‖u‖ (L2 norm)")
    plt.title("Action norms vs. noise (behavioral smoothness)")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[Plot] Saved action norms vs noise -> {out_path}")


def plot_robustness_slopes(slopes_df: pd.DataFrame, out_path: str) -> None:
    """
    Bar plot of slope of return vs noise, grouped by (agent_type, seed).
    Negative slope = performance drops as noise increases.
    """
    if slopes_df.empty:
        print("[Plot] No robustness slopes to plot (need ≥2 noise levels per agent/seed).")
        return

    ensure_dir(out_path)
    plt.figure(figsize=(7, 4))

    labels = [
        f"{row.agent_type}-seed{row.seed}" for _, row in slopes_df.iterrows()
    ]
    x = np.arange(len(labels))
    y = slopes_df["slope"].values

    plt.bar(x, y)
    plt.axhline(0, color="black", linewidth=1)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Slope of Return vs σ")
    plt.xlabel("Agent / seed")
    plt.title("Robustness: performance degradation under noise")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[Plot] Saved robustness slope plot -> {out_path}")


# ---------- Plotting functions (TRAINING, if available) ----------

def plot_training_return(metrics_df: pd.DataFrame, out_path: str) -> None:
    """
    Expect columns: 'envSteps' and 'totalReward'.
    """
    if not {"envSteps", "totalReward"}.issubset(metrics_df.columns):
        print("[Train] metrics CSV has no envSteps/totalReward; "
              "skipping training performance plot.")
        return

    ensure_dir(out_path)
    plt.figure(figsize=(6, 4))

    plt.plot(
        metrics_df["envSteps"].values,
        metrics_df["totalReward"].values,
        "-",
    )
    plt.xlabel("Environment steps")
    plt.ylabel("Return (training eval)")
    plt.title("Task performance vs environment steps")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[Plot] Saved training return vs env steps -> {out_path}")


def plot_sample_efficiency(
    metrics_df: pd.DataFrame,
    reward_threshold: float,
    out_path: str,
) -> None:
    """
    For the single run in metrics_df, find earliest envSteps where totalReward >= threshold.
    Plot reward vs steps and highlight that point.
    """
    if not {"envSteps", "totalReward"}.issubset(metrics_df.columns):
        print("[Train] metrics CSV has no envSteps/totalReward; "
              "skipping sample efficiency plot.")
        return

    ensure_dir(out_path)
    steps = metrics_df["envSteps"].values
    rews = metrics_df["totalReward"].values

    # Find index where reward first crosses threshold
    idx = np.where(rews >= reward_threshold)[0]
    hit = len(idx) > 0
    plt.figure(figsize=(6, 4))
    plt.plot(steps, rews, "-", label="Return")

    plt.axhline(
        reward_threshold,
        color="red",
        linestyle="--",
        label=f"Threshold = {reward_threshold}",
    )

    if hit:
        first_idx = idx[0]
        plt.scatter(
            steps[first_idx],
            rews[first_idx],
            color="green",
            zorder=5,
            label=f"Reached at {steps[first_idx]:.0f} env steps",
        )
        print(
            f"[Train] Sample efficiency: threshold {reward_threshold} "
            f"reached at {steps[first_idx]} env steps."
        )
    else:
        print(
            f"[Train] Threshold {reward_threshold} never reached "
            f"(max reward = {rews.max():.2f})."
        )

    plt.xlabel("Environment steps")
    plt.ylabel("Return (training eval)")
    plt.title("Sample efficiency (threshold crossing)")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[Plot] Saved sample efficiency plot -> {out_path}")


def plot_model_quality(metrics_df: pd.DataFrame, out_path: str) -> None:
    """
    Proxy for model quality: plot worldModelLoss (or reconstructionLoss, etc.)
    vs gradientSteps if available.
    """
    candidate_loss_cols = [
        "worldModelLoss",
        "reconstructionLoss",
        "rewardPredictorLoss",
        "klLoss",
        "model_error",
    ]
    loss_col = None
    for c in candidate_loss_cols:
        if c in metrics_df.columns:
            loss_col = c
            break

    if loss_col is None or "gradientSteps" not in metrics_df.columns:
        print("[Train] No suitable model quality column found; "
              "skipping model-quality plot.")
        return

    ensure_dir(out_path)
    plt.figure(figsize=(6, 4))
    plt.plot(
        metrics_df["gradientSteps"].values,
        metrics_df[loss_col].values,
        "-",
    )
    plt.xlabel("Gradient steps")
    plt.ylabel(loss_col)
    plt.title(f"Model quality proxy: {loss_col} vs gradient steps")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[Plot] Saved model-quality plot ({loss_col}) -> {out_path}")


# ---------- Main entry ----------

def main():
    parser = argparse.ArgumentParser(
        description="Visualization for Cartpole Dreamer experiments."
    )
    parser.add_argument(
        "--eval_csv",
        type=str,
        default=DEFAULT_EVAL_CSV,
        help="Path to eval CSV generated by cartpole_testing.py.",
    )
    parser.add_argument(
        "--metrics_csv",
        type=str,
        default=DEFAULT_METRICS_CSV,
        help="Path to training metrics CSV (from main.py). "
             "If missing, training plots are skipped.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="plots",
        help="Directory to save all plots.",
    )
    parser.add_argument(
        "--reward_threshold",
        type=float,
        default=800.0,
        help="Reward threshold for sample efficiency plot.",
    )
    args = parser.parse_args()

    # ---- Load data ----
    print(f"[Eval] Loading eval CSV from {args.eval_csv}")
    df_eval = load_eval_csv(args.eval_csv)
    summary = summarize_by_noise(df_eval)
    print("[Eval] Summary by noise:")
    print(summary)

    # ---- Eval plots ----
    plot_return_vs_noise(
        summary, os.path.join(args.out_dir, "return_vs_noise.png")
    )
    plot_failure_rate_vs_noise(
        summary, os.path.join(args.out_dir, "failure_rate_vs_noise.png")
    )
    plot_action_norms_vs_noise(
        summary, os.path.join(args.out_dir, "action_norms_vs_noise.png")
    )

    slopes_df = compute_robustness_slopes(df_eval)
    print("[Eval] Robustness slopes:")
    print(slopes_df)
    plot_robustness_slopes(
        slopes_df, os.path.join(args.out_dir, "robustness_slope.png")
    )

    # ---- Training-based plots (optional) ----
    metrics_df = safe_load_training_csv(args.metrics_csv)
    if metrics_df is not None:
        plot_training_return(
            metrics_df,
            os.path.join(args.out_dir, "train_return_vs_env_steps.png"),
        )
        plot_sample_efficiency(
            metrics_df,
            reward_threshold=args.reward_threshold,
            out_path=os.path.join(
                args.out_dir, "sample_efficiency_threshold.png"
            ),
        )
        plot_model_quality(
            metrics_df,
            os.path.join(args.out_dir, "model_quality.png"),
        )


if __name__ == "__main__":
    main()
