
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



TRAIN_CSV = "Train results/CarRacing-v3_CarRacingV3-Present-1.csv"
EVAL_CSV = "eval_results/eval_carracing.csv"
OUT_DIR = "plots_carracing"

SMOOTH_WINDOW = 25          # smoothing window for training curves
SAMPLE_EFF_THRESHOLD = 500  # threshold for sample efficiency & task performance



def smooth_series(y, window):
    """Simple moving average."""
    if window <= 1:
        return y
    y = np.asarray(y, dtype=float)
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(y, kernel, mode="same")


def load_train_csv(path):
    if not os.path.exists(path):
        print(f"[Training] CSV not found at '{path}'. Skipping training plots.")
        return None
    df = pd.read_csv(path)
    print(f"[Training] Loaded {len(df)} rows from {path}")
    print("[Training] Columns:", list(df.columns))
    return df


def load_eval_csv(path):
    if not os.path.exists(path):
        print(f"[Eval] CSV not found at '{path}'. Skipping eval plots.")
        return None
    df = pd.read_csv(path)
    print(f"[Eval] Loaded {len(df)} rows from {path}")
    print("[Eval] Columns:", list(df.columns))
    return df


#training plots 

def plot_model_quality(train_df, out_dir):
    steps = train_df["envSteps"].to_numpy()
    loss = train_df["worldModelLoss"].to_numpy()
    loss_smooth = smooth_series(loss, SMOOTH_WINDOW)

    plt.figure(figsize=(6, 4))
    plt.plot(steps, loss_smooth)
    plt.xlabel("Environment Steps")
    plt.ylabel("World-Model Loss")
    plt.title("CarRacing-v3: Model Quality (World-Model Loss)")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "carracing_model_quality.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("[Training] Saved:", out_path)


def find_sample_efficiency_step(steps, rewards_smooth, threshold):
    """First env step where smoothed reward crosses threshold."""
    above = np.where(rewards_smooth >= threshold)[0]
    if len(above) == 0:
        return None
    idx = int(above[0])
    return int(steps[idx])


def plot_sample_efficiency(train_df, out_dir):
    steps = train_df["envSteps"].to_numpy()
    rewards = train_df["totalReward"].to_numpy()
    rewards_smooth = smooth_series(rewards, SMOOTH_WINDOW)

    step_hit = find_sample_efficiency_step(steps, rewards_smooth,
                                           SAMPLE_EFF_THRESHOLD)

    plt.figure(figsize=(6, 4))
    plt.plot(steps, rewards_smooth, label="Smoothed return")
    plt.axhline(SAMPLE_EFF_THRESHOLD,
                linestyle="--", linewidth=1.2,
                label=f"Threshold = {SAMPLE_EFF_THRESHOLD:.1f}")
    if step_hit is not None:
        plt.axvline(step_hit, linestyle="--", linewidth=1.2,
                    label=f"Reached at {step_hit} steps")

    plt.xlabel("Environment Steps")
    plt.ylabel("Total Reward")
    plt.title("CarRacing-v3: Sample Efficiency")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, "carracing_sample_efficiency.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("[Training] Saved:", out_path)


def plot_task_performance(train_df, out_dir):
    steps = train_df["envSteps"].to_numpy()
    rewards = train_df["totalReward"].to_numpy()
    rewards_smooth = smooth_series(rewards, SMOOTH_WINDOW)

    step_hit = find_sample_efficiency_step(steps, rewards_smooth,
                                           SAMPLE_EFF_THRESHOLD)

    plt.figure(figsize=(6, 4))
    plt.plot(steps, rewards, alpha=0.2, label="Episode return")
    plt.plot(steps, rewards_smooth, label="Smoothed return", linewidth=2.0)
    plt.axhline(SAMPLE_EFF_THRESHOLD,
                linestyle="--", linewidth=1.2,
                label=f"Threshold = {SAMPLE_EFF_THRESHOLD:.1f}")
    if step_hit is not None:
        plt.axvline(step_hit, linestyle="--", linewidth=1.2,
                    label=f"Reached at {step_hit} steps")

    plt.xlabel("Environment Steps")
    plt.ylabel("Total Reward")
    plt.title("CarRacing-v3: Task Performance (Return vs Env Steps)")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, "carracing_task_performance.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("[Training] Saved:", out_path)


#best checkpoint

def compute_best_checkpoint(eval_df):
    """
    Choose best checkpoint = argmax mean return at noise_sigma = 0.
    If σ=0 is missing, fall back to latest checkpoint_step.
    """
    clean = eval_df[eval_df["noise_sigma"] == 0.0]
    if len(clean) == 0:
        print("[Eval] No σ=0 rows. Falling back to last checkpoint.")
        best_step = int(eval_df["checkpoint_step"].max())
        return best_step

    grouped = clean.groupby("checkpoint_step")["return"].mean()
    best_step = int(grouped.idxmax())
    best_ret = float(grouped.max())
    print(f"[Eval] Best checkpoint (σ=0) = {best_step} steps "
          f"(mean return = {best_ret:.1f})")
    return best_step


def plot_best_checkpoint_robustness(eval_df, out_dir, best_step):
    # Filter to chosen checkpoint
    best_df = eval_df[eval_df["checkpoint_step"] == best_step].copy()
    grouped = best_df.groupby("noise_sigma")

    sigmas = sorted(grouped.groups.keys())
    n_per_sigma = np.array([len(grouped.get_group(s)) for s in sigmas])

    # -------- Performance vs Noise --------
    mean_returns = np.array(
        [grouped.get_group(s)["return"].mean() for s in sigmas]
    )
    std_returns = np.array(
        [grouped.get_group(s)["return"].std(ddof=1) for s in sigmas]
    )
    ci95_returns = 1.96 * std_returns / np.sqrt(n_per_sigma)

    plt.figure(figsize=(6, 4))
    plt.errorbar(sigmas, mean_returns, yerr=ci95_returns,
                 marker="o", capsize=3)
    plt.xlabel("Noise σ")
    plt.ylabel("Mean Return")
    plt.title("Best Checkpoint: Performance vs Noise\n(95% CI over episodes)")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "carracing_best_perf_vs_noise.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("[Eval] Saved:", out_path)

    # -------- Failure Rate vs Noise --------
    fail_rates = []
    for s in sigmas:
        g = grouped.get_group(s)
        rate = (g["failed"].sum() / len(g)) * 100.0
        fail_rates.append(rate)

    plt.figure(figsize=(6, 4))
    plt.plot(sigmas, fail_rates, marker="o")
    plt.xlabel("Noise σ")
    plt.ylabel("Failure Rate (%)")
    plt.title("Best Checkpoint: Failure Rate vs Noise")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "carracing_best_failure_vs_noise.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("[Eval] Saved:", out_path)

    # -------- Action Norms vs Noise --------
    action_means = np.array(
        [grouped.get_group(s)["action_norm_mean"].mean() for s in sigmas]
    )
    action_stds = np.array(
        [grouped.get_group(s)["action_norm_mean"].std(ddof=1) for s in sigmas]
    )
    ci95_actions = 1.96 * action_stds / np.sqrt(n_per_sigma)

    plt.figure(figsize=(6, 4))
    plt.errorbar(sigmas, action_means, yerr=ci95_actions,
                 marker="o", capsize=3)
    plt.xlabel("Noise σ")
    plt.ylabel("Action Magnitude")
    plt.title("Best Checkpoint: Action Norms vs Noise\n(95% CI over episodes)")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "carracing_best_action_norms_vs_noise.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("[Eval] Saved:", out_path)

#trajectories vs checkpoint

def plot_trajectories_vs_checkpoint(eval_df, out_dir):
    sigmas = sorted(eval_df["noise_sigma"].unique())

    # --- Return vs checkpoint ---
    plt.figure(figsize=(8, 6))
    for sigma in sigmas:
        sub = eval_df[eval_df["noise_sigma"] == sigma]
        grouped = sub.groupby("checkpoint_step")["return"]
        steps = np.array(sorted(grouped.groups.keys()))
        means = np.array([grouped.get_group(k).mean() for k in steps])
        stds = np.array([grouped.get_group(k).std(ddof=1) for k in steps])
        n = np.array([len(grouped.get_group(k)) for k in steps])
        ci95 = 1.96 * stds / np.sqrt(n)

        plt.errorbar(steps, means, yerr=ci95, marker="o", capsize=3,
                     label=f"σ={sigma:.2f}")

    plt.xlabel("Checkpoint Step")
    plt.ylabel("Mean Return")
    plt.title("CarRacing-v3 Trajectory: Return vs Checkpoint\n"
              "σ ∈ {0.00, 0.02, ..., 0.10}")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, "carracing_traj_return.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("[Eval] Saved:", out_path)

    # --- Failure rate vs checkpoint ---
    plt.figure(figsize=(8, 6))
    for sigma in sigmas:
        sub = eval_df[eval_df["noise_sigma"] == sigma]
        grouped = sub.groupby("checkpoint_step")
        steps = np.array(sorted(grouped.groups.keys()))
        fail_rates = []
        for k in steps:
            g = grouped.get_group(k)
            fail_rates.append((g["failed"].sum() / len(g)) * 100.0)
        plt.plot(steps, fail_rates, marker="o", label=f"σ={sigma:.2f}")

    plt.xlabel("Checkpoint Step")
    plt.ylabel("Failure Rate (%)")
    plt.title("CarRacing-v3 Trajectory: Failure Rate vs Checkpoint\n"
              "σ ∈ {0.00, 0.02, ..., 0.10}")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, "carracing_traj_failure.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("[Eval] Saved:", out_path)

    # --- Action norms vs checkpoint ---
    plt.figure(figsize=(8, 6))
    for sigma in sigmas:
        sub = eval_df[eval_df["noise_sigma"] == sigma]
        grouped = sub.groupby("checkpoint_step")["action_norm_mean"]
        steps = np.array(sorted(grouped.groups.keys()))
        means = np.array([grouped.get_group(k).mean() for k in steps])

        plt.plot(steps, means, marker="o", label=f"σ={sigma:.2f}")

    plt.xlabel("Checkpoint Step")
    plt.ylabel("Action Magnitude")
    plt.title("CarRacing-v3 Trajectory: Action Norms vs Checkpoint\n"
              "σ ∈ {0.00, 0.02, ..., 0.10}")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, "carracing_traj_action_norms.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("[Eval] Saved:", out_path)




def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ----- Training plots -----
    train_df = load_train_csv(TRAIN_CSV)
    if train_df is not None:
        plot_model_quality(train_df, OUT_DIR)
        plot_sample_efficiency(train_df, OUT_DIR)
        plot_task_performance(train_df, OUT_DIR)

    # ----- Eval plots -----
    eval_df = load_eval_csv(EVAL_CSV)
    if eval_df is not None:
        best_step = compute_best_checkpoint(eval_df)
        plot_best_checkpoint_robustness(eval_df, OUT_DIR, best_step)
        plot_trajectories_vs_checkpoint(eval_df, OUT_DIR)


if __name__ == "__main__":
    main()
