#!/usr/bin/env python3
"""
Extract best and final loss values from pendulum training CSV
"""
import pandas as pd
import os

# Path to the pendulum training results CSV
csv_path = "/Users/maryam/Desktop/Fall2025/Deep_Learning/NaturalDreamer/Train results/pendulum-swingup_PendulumSwingup-Present-1.csv"

# Read the CSV file
df = pd.read_csv(csv_path)

# Display available columns
print("Available columns:")
print(df.columns.tolist())
print()

# Loss columns to analyze
loss_columns = [
    'worldModelLoss',
    'reconstructionLoss', 
    'rewardPredictorLoss',
    'klLoss',
    'actorLoss',
    'criticLoss'
]

print("=" * 80)
print("PENDULUM TRAINING LOSS ANALYSIS")
print("=" * 80)
print()

# For each loss column, show best (minimum) and final values
for col in loss_columns:
    if col in df.columns:
        best_loss = df[col].min()
        final_loss = df[col].iloc[-1]
        best_idx = df[col].idxmin()
        best_step = df.loc[best_idx, 'gradientSteps']
        
        print(f"{col}:")
        print(f"  Best Loss:  {best_loss:.6f} (at step {int(best_step)})")
        print(f"  Final Loss: {final_loss:.6f}")
        print()

# Also show summary statistics
print("=" * 80)
print("OVERALL LOSS STATISTICS")
print("=" * 80)
print()

for col in loss_columns:
    if col in df.columns:
        print(f"{col}:")
        print(f"  Mean:     {df[col].mean():.6f}")
        print(f"  Std:      {df[col].std():.6f}")
        print(f"  Min:      {df[col].min():.6f}")
        print(f"  Max:      {df[col].max():.6f}")
        print()

# Show total reward progress
print("=" * 80)
print("TOTAL REWARD PROGRESS")
print("=" * 80)
print()
print(f"Initial Reward:  {df['totalReward'].iloc[0]:.2f}")
print(f"Final Reward:    {df['totalReward'].iloc[-1]:.2f}")
print(f"Best Reward:     {df['totalReward'].max():.2f}")
print(f"Mean Reward:     {df['totalReward'].mean():.2f}")
