import os
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIG –- change this to the actual Pendulum train CSV path
# ============================================================

TRAIN_CSV = "Train results/pendulum-swingup_PendulumSwingup-Present-1.csv"   # <-- EDIT ME
OUT_DIR   = "plots_pendulum"

# Threshold to visualize when “good enough” performance is reached.
# You can tweak this (e.g. 200, 300, 400…) based on your returns.
PERF_THRESHOLD = 200.0

# Moving-average window (in episodes) for smoothing returns
SMOOTH_WINDOW = 50


# ============================================================
# HELPERS
# ============================================================

def load_train_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"TRAIN_CSV not found: {path}")

    df = pd.read_csv(path)
    print("[Train] Loaded CSV with columns:", list(df.columns))

    # Basic sanity checks
    required_cols = ["envSteps", "totalReward", "worldModelLoss"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(
                f"Expected column '{col}' in {path} but only found "
                f"{list(df.columns)}"
            )
    return df


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average, same-length output."""
    if window <= 1:
        return x
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(x, kernel, mode="same")


def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ============================================================
# PLOTS
# ============================================================

def plot_world_model_loss(df: pd.DataFrame, out_dir: str) -> None:
    steps = df["envSteps"].to_numpy()
    loss  = df["worldModelLoss"].to_numpy()

    plt.figure(figsize=(8, 5))
    plt.plot(steps, loss)
    plt.title("Pendulum Swingup: Model Quality (World-Model Loss)")
    plt.xlabel("Environment Steps")
    plt.ylabel("World-Model Loss")
    plt.tight_layout()

    out_path = os.path.join(out_dir, "pendulum_model_quality.png")
    plt.savefig(out_path)
    plt.close()
    print("[Plot] Saved:", out_path)


def plot_sample_efficiency(df: pd.DataFrame,
                           out_dir: str,
                           threshold: float,
                           smooth_window: int) -> None:
    steps = df["envSteps"].to_numpy()
    returns = df["totalReward"].to_numpy()

    smooth_returns = moving_average(returns, smooth_window)

    # Find first step where smoothed return crosses threshold
    idx = np.where(smooth_returns >= threshold)[0]
    if len(idx) > 0:
        hit_idx = int(idx[0])
        hit_step = int(steps[hit_idx])
        hit_return = float(smooth_returns[hit_idx])
        reached_text = f"Reached at {hit_step} steps"
    else:
        hit_idx = None
        hit_step = None
        hit_return = None
        reached_text = "Threshold not reached"

    plt.figure(figsize=(8, 5))
    # raw noisy rewards
    plt.plot(steps, returns, alpha=0.25, label="Episode return")
    # smoothed rewards
    plt.plot(steps, smooth_returns, label="Smoothed return")

    # horizontal threshold line
    plt.axhline(threshold, linestyle="--", label=f"Threshold = {threshold:.1f}")

    # vertical line where threshold is first hit
    if hit_idx is not None:
        plt.axvline(hit_step, linestyle="--", label=reached_text)

    plt.title("Pendulum Swingup: Task Performance (Return vs Env Steps)")
    plt.xlabel("Environment Steps")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(out_dir, "pendulum_task_performance.png")
    plt.savefig(out_path)
    plt.close()
    print("[Plot] Saved:", out_path)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    ensure_outdir(OUT_DIR)
    train_df = load_train_csv(TRAIN_CSV)

    plot_world_model_loss(train_df, OUT_DIR)
    plot_sample_efficiency(train_df, OUT_DIR,
                           threshold=PERF_THRESHOLD,
                           smooth_window=SMOOTH_WINDOW)


if __name__ == "__main__":
    main()
