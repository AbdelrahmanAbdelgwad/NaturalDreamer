#!/bin/bash

# Script to run Lyapunov-enhanced Dreamer experiments

echo "==================================="
echo "Lyapunov-Enhanced Dreamer Training"
echo "==================================="

# Check if CUDA is available
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
echo ""

# Parse command line arguments
MODE=${1:-"lyapunov"}  # Options: baseline, lyapunov, both, ablation
STEPS=${2:-100000}      # Number of gradient steps

# Function to run baseline Dreamer
run_baseline() {
    echo "Running Baseline Dreamer..."
    echo "------------------------"
    python main.py --config cartpole.yml
}

# Function to run Lyapunov Dreamer
run_lyapunov() {
    echo "Running Dreamer with Lyapunov Regularization..."
    echo "---------------------------------------------"
    # First ensure the imports are correct in the Lyapunov version
    sed -i 's/from networks import/from networks_lyapunov import LyapunovModel\nfrom networks import/g' dreamer_lyapunov.py 2>/dev/null || \
    sed -i '' 's/from networks import/from networks_lyapunov import LyapunovModel\nfrom networks import/g' dreamer_lyapunov.py
    
    python main_lyapunov.py --config cartpole_lyapunov.yml
}

# Function to run ablation study
run_ablation() {
    echo "Running Ablation Study..."
    echo "------------------------"
    
    # Create ablation configs
    python compare_experiments.py --mode ablation
    
    # Run experiments with different lambda values
    for lambda in 0.0 0.01 0.05 0.1 0.2 0.5; do
        echo ""
        echo "Running with lambda=$lambda"
        echo "========================="
        python main_lyapunov.py --config cartpole_lambda_${lambda}.yml
    done
}

# Function to compare results
compare_results() {
    echo "Comparing Results..."
    echo "-------------------"
    python compare_experiments.py --mode compare
    echo "Comparison plot saved to comparison_plot.html"
}

# Main execution
case $MODE in
    baseline)
        run_baseline
        ;;
    lyapunov)
        run_lyapunov
        ;;
    both)
        run_baseline
        echo ""
        echo "==================================="
        echo ""
        run_lyapunov
        echo ""
        echo "==================================="
        echo ""
        compare_results
        ;;
    ablation)
        run_ablation
        ;;
    compare)
        compare_results
        ;;
    *)
        echo "Usage: $0 [baseline|lyapunov|both|ablation|compare] [num_steps]"
        echo ""
        echo "Options:"
        echo "  baseline  - Run baseline Dreamer without Lyapunov"
        echo "  lyapunov  - Run Dreamer with Lyapunov regularization"
        echo "  both      - Run both and compare"
        echo "  ablation  - Run ablation study with different lambda values"
        echo "  compare   - Compare existing results"
        exit 1
        ;;
esac

echo ""
echo "Training complete!"
echo ""

# Display results location
echo "Results saved to:"
echo "  Metrics: metrics_lyapunov/*.csv"
echo "  Plots: plots_lyapunov/*.html"
echo "  Checkpoints: checkpoints_lyapunov/*.pth"
echo "  Videos: videos_lyapunov/*.mp4"
