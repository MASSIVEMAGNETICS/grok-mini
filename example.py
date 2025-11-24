#!/usr/bin/env python3
"""
Simple example script to run Grok-Mini V2
"""

from grok_mini import GrokMiniV2, generate, config

def main():
    print("=" * 60)
    print("Grok-Mini V2 Example")
    print("=" * 60)
    
    # Initialize model
    print(f"\nInitializing model on {config.device}...")
    model = GrokMiniV2().to(config.device).to(config.dtype)
    
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model loaded successfully!")
    print(f"Parameters: {param_count:.1f}M")
    print(f"Device: {config.device}")
    print(f"Dtype: {config.dtype}")
    
    # Test text generation
    print("\n" + "=" * 60)
    print("Text Generation Test")
    print("=" * 60)
    
    prompts = [
        "What is artificial intelligence?",
        "Explain machine learning in simple terms:",
        "The future of AI is"
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        print("-" * 40)
        response = generate(model, prompt, max_new_tokens=50, temperature=0.7)
        print(f"Response: {response}")
        print()

if __name__ == "__main__":
    main()
