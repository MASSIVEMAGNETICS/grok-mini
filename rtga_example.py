#!/usr/bin/env python3
"""
Example script demonstrating the Recursive Tool-Graph Agent (RTGA)

NOTE: This requires an OpenAI API key to be set:
    export OPENAI_API_KEY="sk-..."

This example demonstrates:
1. Tool generation from natural language
2. Tool storage in semantic graph
3. Zero-shot tool retrieval
4. Graph visualization
"""

import os
import sys

def main():
    # Check for API key before proceeding
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable is required")
        print("Please set it before running:")
        print("    export OPENAI_API_KEY='sk-...'")
        sys.exit(1)
    
    from recursive_tool_agent import RecursiveBuilder, ConsoleStyle
    
    print(f"\n{ConsoleStyle.BOLD}=== MASSIVE MAGNETICS RTGA DEMO ==={ConsoleStyle.RESET}\n")
    
    # Initialize the agent
    bot = RecursiveBuilder()
    
    # Demo 1: Generate a math tool
    print("\n" + "=" * 60)
    print("DEMO 1: Generate a Fibonacci Calculator")
    print("=" * 60)
    bot.execute("Write a function to calculate the fibonacci sequence")
    
    # Demo 2: Generate a text processing tool
    print("\n" + "=" * 60)
    print("DEMO 2: Generate a Password Generator")
    print("=" * 60)
    bot.execute("Create a function to generate a secure random password")
    
    # Demo 3: Generate a data structure tool
    print("\n" + "=" * 60)
    print("DEMO 3: Generate a List Sorter")
    print("=" * 60)
    bot.execute("Create a function to sort a list of numbers")
    
    # Demo 4: Retrieve a previously generated tool (The Hook!)
    print("\n" + "=" * 60)
    print("DEMO 4: Zero-Shot Tool Retrieval")
    print("=" * 60)
    print("Attempting to retrieve the fibonacci calculator...")
    bot.execute("Run the calculate_fibonacci function")
    
    # Display graph statistics
    print("\n" + "=" * 60)
    print("GRAPH STATISTICS")
    print("=" * 60)
    print(f"Total nodes: {bot.memory.graph.number_of_nodes()}")
    print(f"Total edges: {bot.memory.graph.number_of_edges()}")
    print(f"Tools stored: {len(bot.memory.tools)}")
    print(f"Tool names: {list(bot.memory.tools.keys())}")
    
    # Visualize the tool graph
    print("\n" + "=" * 60)
    print("GRAPH VISUALIZATION")
    print("=" * 60)
    print("Generating visualization...")
    print("(Close the plot window to continue)")
    
    try:
        bot.visualize()
    except Exception as e:
        print(f"Visualization skipped: {e}")
        print("Note: This requires a display. Run on a system with GUI support.")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("\nThe RTGA has demonstrated:")
    print("  ✓ Tool generation from natural language")
    print("  ✓ Semantic categorization and storage")
    print("  ✓ Zero-shot tool retrieval")
    print("  ✓ Graph-based memory persistence")
    print("\nAll generated tools remain in memory for future use!")

if __name__ == "__main__":
    main()
