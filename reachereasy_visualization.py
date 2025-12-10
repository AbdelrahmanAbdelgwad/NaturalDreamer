import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TRAIN_CSV = "Train results/reacher-easy_ReacherEasy-2.csv"
EVAL_CSV = "eval_results/eval_reacher.csv"
OUT_DIR = "plots_reacher"

SAMPLE_EFF_THRESHOLD = 900.0 

def ensure_dir(path: str) -> None:
    d = os.path.dirname(path) if os.path.splitext(path)[1] else path
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def mean_and_ci(x: pd.Series, alpha: float = 0.95):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return np.nan, np.nan
    mean = x.mean()
    if n == 1:
        return mean, 0.0
    std = x.std(ddof=1)
    z = 1.96  # 95%
    ci = z * std / np.sqrt(n)
    return mean, ci


ensure_dir(OUT_DIR)

train = pd.read_csv(TRAIN_CSV)
eval_df = pd.read_csv(EVAL_CSV)



# Plot 1: Task performance (Return vs env steps)
plt.figure(figsize=(8, 6))
plt.plot(train["envSteps"], train["totalReward"])
plt.xlabel("Environment Steps")
plt.ylabel("Total Reward")
plt.title("Reacher-Easy: Task Performance (Return vs Environment Steps)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "reacher_task_performance.png"))

# Plot 2: Sample efficiency (steps to reach threshold)
mask = train["totalReward"] >= SAMPLE_EFF_THRESHOLD
if mask.any():
    step_to_threshold = int(train.loc[mask, "envSteps"].iloc[0])
else:
    step_to_threshold = None

plt.figure(figsize=(8, 6))
plt.plot(train["envSteps"], train["totalReward"])
plt.axhline(SAMPLE_EFF_THRESHOLD, linestyle="--")
if step_to_threshold is not None:
    plt.axvline(step_to_threshold, linestyle=":")
    subtitle = f"Reached at {step_to_threshold:,} steps"
else:
    subtitle = "Threshold not reached"
plt.xlabel("Environment Steps")
plt.ylabel("Total Reward")
plt.title(
    f"Reacher-Easy: Sample Efficiency (threshold={SAMPLE_EFF_THRESHOLD})\n{subtitle}"
)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "reacher_sample_efficiency.png"))

# Plot 3: Model quality (world-model loss vs env steps, smoothed)
wm_loss = train["worldModelLoss"].rolling(window=500, min_periods=1).mean()
plt.figure(figsize=(8, 6))
plt.plot(train["envSteps"], wm_loss)
plt.xlabel("Environment Steps")
plt.ylabel("World Model Loss (smoothed)")
plt.title("Reacher-Easy: Model Quality (World-Model Prediction Loss)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "reacher_model_quality.png"))


# a) Find best checkpoint: highest mean return at sigma = 0
noise0 = eval_df[np.isclose(eval_df["noise_sigma"], 0.0)]
best_group = noise0.groupby("checkpoint_step")["return"].mean()
best_step = int(best_group.idxmax())
best_eval = eval_df[eval_df["checkpoint_step"] == best_step]

print(f"Best checkpoint (Reacher-Easy) is step={best_step}")


sigmas = np.sort(eval_df["noise_sigma"].unique())

# ---- Plot 4: Best checkpoint – Performance vs noise ----
group = best_eval.groupby("noise_sigma")["return"]
means = group.mean()
cis = group.apply(lambda x: mean_and_ci(x)[1])

perf_stats = pd.DataFrame(
    {"noise_sigma": means.index.values, "mean": means.values, "ci": cis.values}
)

plt.figure(figsize=(8, 6))
plt.errorbar(
    perf_stats["noise_sigma"],
    perf_stats["mean"],
    yerr=perf_stats["ci"],
    fmt="o-",
)
plt.xlabel("Noise σ")
plt.ylabel("Mean Return")
plt.title(
    f"Best Checkpoint (step={best_step}): Performance vs Noise\n(95% CI over episodes)"
)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "reacher_best_perf_vs_noise.png"))

# ---- Plot 5: Best checkpoint – Action norms vs noise ----
group = best_eval.groupby("noise_sigma")["action_norm_mean"]
an_means = group.mean()
an_cis = group.apply(lambda x: mean_and_ci(x)[1])

an_stats = pd.DataFrame(
    {"noise_sigma": an_means.index.values, "mean": an_means.values, "ci": an_cis.values}
)

plt.figure(figsize=(8, 6))
plt.errorbar(
    an_stats["noise_sigma"],
    an_stats["mean"],
    yerr=an_stats["ci"],
    fmt="o-",
)
plt.xlabel("Noise σ")
plt.ylabel("Action Magnitude")
plt.title(
    f"Best Checkpoint (step={best_step}): Action Norms vs Noise\n(95% CI over episodes)"
)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "reacher_best_action_norms_vs_noise.png"))

# ---- Plot 6: Best checkpoint – Failure rate vs noise ----
fail_rates = (
    best_eval.groupby("noise_sigma")["failed"].mean() * 100.0
)  # % of episodes failed

plt.figure(figsize=(8, 6))
plt.plot(fail_rates.index.values, fail_rates.values, marker="o")
plt.xlabel("Noise σ")
plt.ylabel("Failure Rate (%)")
plt.title(f"Best Checkpoint (step={best_step}): Failure Rate vs Noise")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "reacher_best_failure_vs_noise.png"))



# ---- Plot 7: Trajectory – Return vs checkpoint (per sigma) ----
plt.figure(figsize=(8, 6))
for sigma in sigmas:
    sub = eval_df[np.isclose(eval_df["noise_sigma"], sigma)]
    group = sub.groupby("checkpoint_step")["return"]
    means = group.mean()
    cis = group.apply(lambda x: mean_and_ci(x)[1])

    stats = pd.DataFrame(
        {
            "checkpoint_step": means.index.values,
            "mean": means.values,
            "ci": cis.values,
        }
    )

    plt.errorbar(
        stats["checkpoint_step"],
        stats["mean"],
        yerr=stats["ci"],
        marker="o",
        label=f"σ={sigma:.2f}",
    )

plt.xlabel("Checkpoint Step")
plt.ylabel("Mean Return")
plt.title(
    "Reacher-Easy Trajectory: Return vs Checkpoint\nσ ∈ "
    + "{" + ", ".join([f"{s:.2f}" for s in sigmas]) + "}"
)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "reacher_traj_return.png"))

# ---- Plot 8: Trajectory – Action norms vs checkpoint (per sigma) ----
plt.figure(figsize=(8, 6))
for sigma in sigmas:
    sub = eval_df[np.isclose(eval_df["noise_sigma"], sigma)]
    group = sub.groupby("checkpoint_step")["action_norm_mean"]
    means = group.mean()
    cis = group.apply(lambda x: mean_and_ci(x)[1])

    stats = pd.DataFrame(
        {
            "checkpoint_step": means.index.values,
            "mean": means.values,
            "ci": cis.values,
        }
    )

    plt.errorbar(
        stats["checkpoint_step"],
        stats["mean"],
        yerr=stats["ci"],
        marker="o",
        label=f"σ={sigma:.2f}",
    )

plt.xlabel("Checkpoint Step")
plt.ylabel("Action Magnitude")
plt.title(
    "Reacher-Easy Trajectory: Action Norms vs Checkpoint\nσ ∈ "
    + "{" + ", ".join([f"{s:.2f}" for s in sigmas]) + "}"
)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "reacher_traj_action_norms.png"))

# ---- Plot 9: Trajectory – Failure rate vs checkpoint (per sigma) ----
plt.figure(figsize=(8, 6))
for sigma in sigmas:
    sub = eval_df[np.isclose(eval_df["noise_sigma"], sigma)]
    fail_rate = sub.groupby("checkpoint_step")["failed"].mean() * 100.0
    plt.plot(
        fail_rate.index.values,
        fail_rate.values,
        marker="o",
        label=f"σ={sigma:.2f}",
    )

plt.xlabel("Checkpoint Step")
plt.ylabel("Failure Rate (%)")
plt.title(
    "Reacher-Easy Trajectory: Failure Rate vs Checkpoint\nσ ∈ "
    + "{" + ", ".join([f"{s:.2f}" for s in sigmas]) + "}"
)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "reacher_traj_failure.png"))

print("All Reacher-Easy plots saved in:", OUT_DIR)
