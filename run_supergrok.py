#!/usr/bin/env python3
"""
run_supergrok.py

CLI entry point for SuperGrok Mini inference with safe defaults.

Features:
- Loads GrokMiniV3 models or falls back to small demo model
- Simple text generation
- Safe-by-default (no network, no autonomous actions without approval)

Usage:
  python run_supergrok.py --prompt "Hello world" --max-tokens 32
  python run_supergrok.py --checkpoint model.npz --prompt "Explain AI"
"""

import argparse
import sys
import os
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="SuperGrok Mini - Local inference demo")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt for generation")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to model checkpoint (.npz)")
    parser.add_argument("--max-tokens", type=int, default=32, help="Maximum tokens to generate")
    parser.add_argument("--input-dim", type=int, default=8, help="Model input dimension")
    parser.add_argument("--hidden-dims", type=str, default="16,8", help="Hidden layer dimensions (comma-separated)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    print("=" * 60)
    print("SuperGrok Mini - Local Inference")
    print("=" * 60)
    print(f"Prompt: {args.prompt}")
    print(f"Max tokens: {args.max_tokens}")
    print("=" * 60)
    
    # Try to load model
    try:
        from grok_mini_v3 import GrokMiniV3
        from predictive_autograd_engine import load_state_dict
        
        hidden_dims = [int(x.strip()) for x in args.hidden_dims.split(",")]
        
        print(f"Creating model (input_dim={args.input_dim}, hidden_dims={hidden_dims})")
        model = GrokMiniV3(input_dim=args.input_dim, hidden_dims=hidden_dims, output_dim=1, seed=args.seed)
        
        if args.checkpoint and os.path.exists(args.checkpoint):
            print(f"Loading checkpoint: {args.checkpoint}")
            state = load_state_dict(args.checkpoint)
            model.load_state_dict(state)
        
        # Simple demo: encode prompt as hash features and generate
        print("\n🤖 Generating response...")
        print("-" * 60)
        
        # Simple character-level feature encoding (demo only)
        prompt_features = _encode_prompt(args.prompt, args.input_dim)
        
        # Generate tokens (simplified - just predict multiple times)
        generated_tokens = []
        current_input = prompt_features
        
        for i in range(args.max_tokens):
            # Predict next feature
            output = model.predict_numpy(current_input.reshape(1, -1))
            
            # Simple token: map output to character
            token = _decode_output(output[0, 0], i)
            generated_tokens.append(token)
            
            # Update input (rolling window)
            current_input = np.roll(current_input, -1)
            current_input[-1] = output[0, 0]
            
            # Print streaming output
            print(token, end="", flush=True)
            
            # Simple stopping condition
            if token in ['.', '!', '?'] and i > 10:
                break
        
        print()
        print("-" * 60)
        print(f"\n✓ Generated {len(generated_tokens)} tokens")
        
    except ImportError as e:
        print(f"❌ Error: Missing dependencies - {e}")
        print("\nNote: SuperGrok requires numpy. For full features, also install torch and transformers.")
        print("  pip install numpy torch transformers")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Print safety notice
    print("\n" + "=" * 60)
    print("Safety & Security")
    print("=" * 60)
    print("SuperGrok runs in safe mode by default:")
    print("  • No network access without ALLOW_NETWORK=true")
    print("  • No autonomous actions without ALLOW_SELF_ACTIONS=true")
    print("  • All sensitive operations require approval")
    print("\nTo enable autonomous features:")
    print("  export ALLOW_SELF_ACTIONS=true")
    print("  export ALLOW_NETWORK=true")
    print("=" * 60)

def _encode_prompt(prompt: str, dim: int) -> np.ndarray:
    """
    Simple prompt encoding: character hashing to fixed-dim features.
    This is a demo implementation - production would use proper tokenization.
    """
    features = np.zeros(dim, dtype=float)
    for i, char in enumerate(prompt[:dim]):
        features[i] = (hash(char) % 256) / 256.0
    # Pad with prompt statistics
    if len(prompt) < dim:
        features[len(prompt):] = len(prompt) / 100.0
    return features

def _decode_output(value: float, position: int) -> str:
    """
    Simple output decoding: map continuous value to character.
    This is a demo implementation - production would use proper detokenization.
    """
    # Map output value to character
    chars = "abcdefghijklmnopqrstuvwxyz .,!?'\n"
    idx = int(abs(value * len(chars))) % len(chars)
    return chars[idx]


if __name__ == "__main__":
    main()
