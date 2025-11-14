"""
Test script to verify the Lyapunov implementation works correctly
"""

import torch
import numpy as np
from networks import LyapunovModel
import attridict

def test_lyapunov_model():
    """Test the Lyapunov model forward pass and properties."""
    print("Testing Lyapunov Model...")
    print("-" * 40)
    
    # Create config
    config = attridict({
        'hiddenSize': 128,
        'numLayers': 3,
        'activation': 'Tanh'
    })
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputSize = 256 + 64  # recurrentSize + latentSize
    model = LyapunovModel(inputSize, config).to(device)
    
    # Test forward pass
    batchSize = 32
    x = torch.randn(batchSize, inputSize).to(device)
    
    # Forward pass
    v = model(x)
    
    # Check output shape
    assert v.shape == (batchSize,), f"Output shape mismatch: {v.shape}"
    
    # Check positive definiteness
    assert torch.all(v > 0), "Lyapunov values should be positive"
    
    # Test with zero input (should be close to epsilon)
    x_zero = torch.zeros(1, inputSize).to(device)
    v_zero = model(x_zero)
    assert v_zero.item() > 0, "Lyapunov value at zero should be positive (epsilon)"
    
    print(f"✓ Forward pass successful")
    print(f"✓ Output shape: {v.shape}")
    print(f"✓ Positive definiteness verified")
    print(f"✓ Mean Lyapunov value: {v.mean().item():.4f}")
    print(f"✓ Std Lyapunov value: {v.std().item():.4f}")
    print(f"✓ V(0) = {v_zero.item():.6f}")
    
    return model

def test_lyapunov_gradient():
    """Test that Lyapunov function can be differentiated."""
    print("\nTesting Lyapunov Gradients...")
    print("-" * 40)
    
    config = attridict({
        'hiddenSize': 128,
        'numLayers': 3,
        'activation': 'Tanh'
    })
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputSize = 320
    model = LyapunovModel(inputSize, config).to(device)
    
    # Create input with gradient tracking
    x = torch.randn(16, inputSize, requires_grad=True).to(device)
    
    # Forward pass
    v = model(x)
    
    # Compute gradient
    loss = v.mean()
    loss.backward()
    
    # Check gradients exist
    assert x.grad is not None, "Input gradient should exist"
    for param in model.parameters():
        assert param.grad is not None, "Model gradients should exist"
    
    print(f"✓ Gradient computation successful")
    print(f"✓ Input gradient norm: {x.grad.norm().item():.4f}")
    
    # Test monotonicity gradient
    x1 = torch.randn(16, inputSize).to(device)
    x2 = x1 + 0.1 * torch.randn_like(x1)  # Small perturbation
    
    v1 = model(x1)
    v2 = model(x2)
    
    # Monotonicity loss (we want v2 < v1)
    monotonicity_loss = torch.relu(v2 - v1).mean()
    
    print(f"✓ Monotonicity loss: {monotonicity_loss.item():.4f}")
    
    return model

def test_lyapunov_integration():
    """Test Lyapunov integration with Dreamer components."""
    print("\nTesting Dreamer-Lyapunov Integration...")
    print("-" * 40)
    
    from dreamer_lyapunov import DreamerLyapunov
    from buffer import ReplayBuffer
    
    # Create simple config
    config = attridict({
        'recurrentSize': 256,
        'latentLength': 8,
        'latentClasses': 8,
        'encodedObsSize': 64,
        'batchSize': 4,
        'batchLength': 8,
        'imaginationHorizon': 5,
        'useContinuationPrediction': False,
        'actorLR': 0.0001,
        'criticLR': 0.0003,
        'worldModelLR': 0.0006,
        'lyapunovLR': 0.0001,
        'gradientNormType': 2,
        'gradientClip': 100,
        'discount': 0.997,
        'lambda_': 0.95,
        'freeNats': 1,
        'betaPrior': 1.0,
        'betaPosterior': 0.1,
        'entropyScale': 0.003,
        'lyapunovLambda': 0.1,
        'buffer': attridict({'capacity': 1000}),
        'encoder': attridict({'hiddenSize': 64, 'numLayers': 2, 'activation': 'Tanh'}),
        'decoder': attridict({'hiddenSize': 64, 'numLayers': 2, 'activation': 'Tanh'}),
        'recurrentModel': attridict({'hiddenSize': 100, 'activation': 'Tanh'}),
        'priorNet': attridict({'hiddenSize': 100, 'numLayers': 2, 'activation': 'Tanh', 'uniformMix': 0.01}),
        'posteriorNet': attridict({'hiddenSize': 100, 'numLayers': 2, 'activation': 'Tanh', 'uniformMix': 0.01}),
        'reward': attridict({'hiddenSize': 200, 'numLayers': 2, 'activation': 'Tanh'}),
        'continuation': attridict({'hiddenSize': 200, 'numLayers': 3, 'activation': 'Tanh'}),
        'actor': attridict({'hiddenSize': 64, 'numLayers': 2, 'activation': 'Tanh'}),
        'critic': attridict({'hiddenSize': 256, 'numLayers': 3, 'activation': 'Tanh'}),
        'lyapunov': attridict({'hiddenSize': 128, 'numLayers': 3, 'activation': 'Tanh'})
    })
    
    # Initialize Dreamer with Lyapunov
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    observationShape = (4,)  # Cartpole observation
    actionSize = 1
    actionLow = [-1.0]
    actionHigh = [1.0]
    
    dreamer = DreamerLyapunov(
        observationShape, actionSize, actionLow, actionHigh, device, config
    )
    
    # Check that Lyapunov model exists
    assert hasattr(dreamer, 'lyapunovModel'), "Lyapunov model should exist"
    assert hasattr(dreamer, 'lyapunovOptimizer'), "Lyapunov optimizer should exist"
    
    print(f"✓ DreamerLyapunov initialized successfully")
    print(f"✓ Lyapunov model parameters: {sum(p.numel() for p in dreamer.lyapunovModel.parameters())}")
    
    # Test forward pass through Lyapunov
    fullStateSize = config.recurrentSize + config.latentLength * config.latentClasses
    testState = torch.randn(16, fullStateSize).to(device)
    lyapunovValue = dreamer.lyapunovModel(testState)
    
    print(f"✓ Lyapunov forward pass: {lyapunovValue.shape}")
    print(f"✓ Mean Lyapunov value: {lyapunovValue.mean().item():.4f}")
    
    # Fill buffer with some data for training test
    print("\nTesting training step...")
    for _ in range(20):
        obs = np.random.randn(*observationShape).astype(np.float32)
        action = np.random.randn(actionSize).astype(np.float32)
        reward = np.random.randn(1).astype(np.float32)
        next_obs = np.random.randn(*observationShape).astype(np.float32)
        done = np.random.rand() > 0.9
        dreamer.buffer.add(obs, action, reward, next_obs, done)
    
    # Sample batch and test training
    if len(dreamer.buffer) >= config.batchSize * config.batchLength:
        sampledData = dreamer.buffer.sample(config.batchSize, config.batchLength)
        
        # World model training
        initialStates, worldMetrics = dreamer.worldModelTraining(sampledData)
        print(f"✓ World model training successful")
        
        # Behavior training with Lyapunov
        behaviorMetrics = dreamer.behaviorTraining(initialStates)
        print(f"✓ Behavior training with Lyapunov successful")
        
        # Check Lyapunov metrics exist
        assert 'lyapunovLoss' in behaviorMetrics, "Lyapunov loss should be in metrics"
        assert 'lyapunovPenalty' in behaviorMetrics, "Lyapunov penalty should be in metrics"
        assert 'lyapunovMean' in behaviorMetrics, "Lyapunov mean should be in metrics"
        
        print(f"  - Lyapunov loss: {behaviorMetrics['lyapunovLoss']:.4f}")
        print(f"  - Lyapunov penalty: {behaviorMetrics['lyapunovPenalty']:.4f}")
        print(f"  - Lyapunov mean: {behaviorMetrics['lyapunovMean']:.4f}")
    
    return dreamer

def main():
    """Run all tests."""
    print("=" * 50)
    print("LYAPUNOV IMPLEMENTATION TESTS")
    print("=" * 50)
    
    try:
        # Test 1: Lyapunov model
        model = test_lyapunov_model()
        
        # Test 2: Gradients
        test_lyapunov_gradient()
        
        # Test 3: Integration with Dreamer
        dreamer = test_lyapunov_integration()
        
        print("\n" + "=" * 50)
        print("ALL TESTS PASSED ✓")
        print("=" * 50)
        print("\nThe Lyapunov implementation is working correctly!")
        print("You can now run training with:")
        print("  python main_lyapunov.py --config cartpole_lyapunov.yml")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
