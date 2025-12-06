import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#  Paths 
TRAIN_CSV = "Train results/cartpole-swingup_CartpoleSwingup-1.csv"
EVAL_CSV = "eval_results/eval_cartpole.csv"

OUT_DIR = "plots"
OUT_BEST = os.path.join(OUT_DIR, "best_checkpoint")
OUT_TRAJ = os.path.join(OUT_DIR, "trajectory")

os.makedirs(OUT_BEST, exist_ok=True)
os.makedirs(OUT_TRAJ, exist_ok=True)

#  Load data 
train = pd.read_csv(TRAIN_CSV)
eval_df = pd.read_csv(EVAL_CSV)

print("Train columns:", list(train.columns))
print("Eval columns:", list(eval_df.columns))

# Some convenience values
noise_vals = np.sort(eval_df["noise_sigma"].unique())
ckpt_vals = np.sort(eval_df["checkpoint_step"].unique())

print("Noise values:", noise_vals)
print("Checkpoint steps:", ckpt_vals)

def mean_and_ci(series: pd.Series):
    x = series.to_numpy(dtype=float)
    n = len(x)
    if n == 0:
        return np.nan, np.nan
    mean = x.mean()
    std = x.std(ddof=1) if n > 1 else 0.0
    ci = 1.96 * std / np.sqrt(max(n, 1))
    return mean, ci


# 1) Task performance vs env steps
plt.figure()
plt.plot(train["envSteps"], train["totalReward"])
plt.xlabel("Environment Steps")
plt.ylabel("Total Reward")
plt.title("Task Performance: Return vs Environment Steps")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "task_performance.png"))

# 2) Sample efficiency: steps to reach threshold
THRESHOLD = 650.0
mask = train["totalReward"] >= THRESHOLD
if mask.any():
    step_reached = train.loc[mask, "envSteps"].iloc[0]
else:
    step_reached = None

plt.figure()
plt.plot(train["envSteps"], train["totalReward"])
plt.axhline(THRESHOLD, linestyle="--", linewidth=1.5)

if step_reached is not None:
    plt.axvline(step_reached, linestyle=":", linewidth=1.5)
    title_extra = f"Reached at {int(step_reached):,} steps"
else:
    title_extra = "Threshold not reached during training"

plt.xlabel("Environment Steps")
plt.ylabel("Total Reward")
plt.title(f"Sample Efficiency: Steps to reach reward {THRESHOLD}\n{title_extra}")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "sample_efficiency.png"))

# 3) Model quality: use worldModelLoss if available
if "worldModelLoss" in train.columns:
    steps = train["envSteps"].values
    loss = train["worldModelLoss"].values
    window = 50
    loss_smooth = (
        pd.Series(loss).rolling(window, min_periods=1).mean().values
    )
    plt.figure()
    plt.plot(steps, loss_smooth)
    plt.xlabel("Environment Steps")
    plt.ylabel("World Model Loss (smoothed)")
    plt.title("Model Quality: World-Model Prediction Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "model_quality.png"))
else:
    print("Model quality plot skipped – 'worldModelLoss' not found in train CSV")

#  BEST CHECKPOINT PLOTS
if "noise_sigma" in eval_df.columns:
    no_noise = eval_df[eval_df["noise_sigma"] == 0.0]
else:
    no_noise = eval_df

best_step = (
    no_noise.groupby("checkpoint_step")["return"]
    .mean()
    .idxmax()
)
print(f"\nBest checkpoint_step (for best-checkpoint plots): {best_step}")

eval_best = eval_df[eval_df["checkpoint_step"] == best_step]
best_grouped = eval_best.groupby("noise_sigma")

# 4) Best checkpoint – failure rate vs noise
fail_rates = []
for sigma, g in best_grouped:
    fail_rate = g["failed"].mean() * 100.0
    fail_rates.append(fail_rate)

plt.figure()
plt.plot(noise_vals, fail_rates, marker="o")
plt.xlabel("Noise σ")
plt.ylabel("Failure Rate (%)")
plt.title(f"Best Checkpoint (step={best_step}): Failure Rate vs Noise")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT_BEST, "failure_rate_vs_noise.png"))

# 5) Best checkpoint – robustness (return vs noise with 95% CI)
means = []
cis = []
for sigma, g in best_grouped:
    m, ci = mean_and_ci(g["return"])
    means.append(m)
    cis.append(ci)

means = np.array(means)
cis = np.array(cis)

plt.figure()
plt.errorbar(noise_vals, means, yerr=cis, marker="o", capsize=4)
plt.xlabel("Noise σ")
plt.ylabel("Mean Return")
plt.title(f"Best Checkpoint (step={best_step}): Performance vs Noise\n(95% CI)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT_BEST, "robustness_vs_noise.png"))

# 6) Best checkpoint – action norms vs noise (95% CI)
act_means = []
act_cis = []
for sigma, g in best_grouped:
    m, ci = mean_and_ci(g["action_norm_mean"])
    act_means.append(m)
    act_cis.append(ci)

act_means = np.array(act_means)
act_cis = np.array(act_cis)

plt.figure()
plt.errorbar(noise_vals, act_means, yerr=act_cis, marker="o", capsize=4)
plt.xlabel("Noise σ")
plt.ylabel("Action Magnitude")
plt.title(f"Best Checkpoint (step={best_step}): Action Norms vs Noise\n(95% CI)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT_BEST, "action_norms_vs_noise.png"))

#  TRAJECTORY PLOTS
desired_sigmas = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10]
sigma_levels = [s for s in desired_sigmas if s in noise_vals]

print(f"\nTrajectory will use noise levels σ = {sigma_levels}")

def metric_vs_checkpoint_for_sigma(df_all, sigma, col_name):
    """Aggregate mean + 95% CI of col_name vs checkpoint_step for a fixed sigma."""
    sub = df_all[df_all["noise_sigma"] == sigma]
    grouped = sub.groupby("checkpoint_step")[col_name]
    means = grouped.mean()
    cis = grouped.apply(mean_and_ci).apply(lambda t: t[1])  # take CI from (mean, ci)
    return means, cis

# 7) Trajectory: Return vs checkpoint for σ in {0.00, 0.02, 0.04, 0.06, 0.08, 0.10}
plt.figure()
for sigma in sigma_levels:
    mean_ret, ci_ret = metric_vs_checkpoint_for_sigma(eval_df, sigma, "return")
    plt.errorbar(
        mean_ret.index,
        mean_ret.values,
        yerr=ci_ret.values,
        marker="o",
        capsize=4,
        label=f"σ={sigma:.2f}",
    )

plt.xlabel("Checkpoint Step")
plt.ylabel("Mean Return")
plt.title("Trajectory: Return vs Checkpoint\nσ ∈ {0.00, 0.02, 0.04, 0.06, 0.08, 0.10}")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_TRAJ, "trajectory_return.png"))

# 8) Trajectory: Failure rate vs checkpoint 
plt.figure()
for sigma in sigma_levels:
    sub = eval_df[eval_df["noise_sigma"] == sigma]
    fail = sub.groupby("checkpoint_step")["failed"].mean() * 100.0
    plt.plot(
        fail.index,
        fail.values,
        marker="o",
        label=f"σ={sigma:.2f}",
    )

plt.xlabel("Checkpoint Step")
plt.ylabel("Failure Rate (%)")
plt.title("Trajectory: Failure Rate vs Checkpoint\nσ ∈ {0.00, 0.02, 0.04, 0.06, 0.08, 0.10}")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_TRAJ, "trajectory_failure.png"))

# 9) Trajectory: Action norms vs checkpoint 
plt.figure()
for sigma in sigma_levels:
    mean_an, ci_an = metric_vs_checkpoint_for_sigma(eval_df, sigma, "action_norm_mean")
    plt.errorbar(
        mean_an.index,
        mean_an.values,
        yerr=ci_an.values,
        marker="o",
        capsize=4,
        label=f"σ={sigma:.2f}",
    )

plt.xlabel("Checkpoint Step")
plt.ylabel("Action Magnitude")
plt.title("Trajectory: Action Norms vs Checkpoint\nσ ∈ {0.00, 0.02, 0.04, 0.06, 0.08, 0.10}")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_TRAJ, "trajectory_action_norms.png"))

# SUMMARY 
print("\n=== Evaluation summary (best checkpoint only) ===")
print(f"Best checkpoint_step: {best_step}")

summary_best = (
    eval_best.groupby("noise_sigma")
    .agg(
        mean_return=("return", "mean"),
        std_return=("return", "std"),
        fail_rate=("failed", "mean"),
        mean_action_norm=("action_norm_mean", "mean"),
    )
)
summary_best["fail_rate"] *= 100.0
print(summary_best)

print(f"\nAll plots saved under '{OUT_DIR}'")
