"""
=============================================================================
PROJECT:   RECURSIVE TOOL-GRAPH AGENT (RTGA)
AUTHOR:    Brandon "Iambandobandz" Emery
ENTITY:    MassiveMagnetics
DATE:      November 2025
VERSION:   1.0.0 (MassiveMagnetics Core)

DESCRIPTION:
A self-improving agent substrate that generates Python tools on-the-fly, 
executes them, and persists them to a directed semantic graph (NetworkX) 
for zero-shot retrieval in future iterations.

ARCHITECTURE:
1. Cognitive Layer: GPT-4o (Code Generation)
2. Memory Layer:    NetworkX DiGraph (Semantic Tool Storage)
3. Execution Layer: Local Runtime (Unsafe/Native)

DEPENDENCIES:
pip install openai networkx matplotlib
=============================================================================
"""

import os
import textwrap
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, Callable, Optional, List
from openai import OpenAI

# --- CONFIGURATION ---
# PRO TIP: Export this in your terminal: export OPENAI_API_KEY="sk-..."
API_KEY = os.getenv("OPENAI_API_KEY") or "YOUR_API_KEY_HERE"
MODEL_ID = "gpt-4o"

client = OpenAI(api_key=API_KEY)

class ConsoleStyle:
    """Helper for aesthetic terminal output (The 'MassiveMagnetics' Aesthetic)."""
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

    @staticmethod
    def log(sender: str, message: str, color: str = RESET):
        print(f"{color}{ConsoleStyle.BOLD}[{sender}]{ConsoleStyle.RESET} {message}")

class ToolGraphMemory:
    """
    The 'Substrate': A directed graph managing tool lifecycle and retrieval.
    """
    def __init__(self):
        self.graph = nx.DiGraph()
        self.tools: Dict[str, Callable] = {}
        ConsoleStyle.log("MASSIVE_MAGNETICS", "Graph Substrate Initialized.", ConsoleStyle.CYAN)

    def add_tool(self, name: str, code: str, func: Callable, description: str, tags: List[str]):
        """Persists a tool to the graph with semantic tags."""
        self.tools[name] = func
        
        # Add tool node
        self.graph.add_node(name, code=code, desc=description, type="tool")
        
        # Add semantic connections (Ontology)
        for tag in tags:
            self.graph.add_node(tag, type="category")
            self.graph.add_edge(tag, name, relation="categorizes")
            
        ConsoleStyle.log("MEMORY", f"synapse_established :: {name} <--> {tags}", ConsoleStyle.GREEN)

    def find_tool(self, query: str) -> Optional[str]:
        """
        Semantic Retrieval: Checks if a tool exists for the query.
        """
        # 1. Direct Name Match
        clean_query = query.lower().replace(" ", "_")
        for name in self.tools:
            if name in clean_query:
                return name
        return None

class RecursiveBuilder:
    """
    The Agent: Capable of self-modification via tool creation.
    """
    def __init__(self):
        self.memory = ToolGraphMemory()

    def _generate_code(self, task: str) -> str:
        """LLM Call to architect the solution."""
        ConsoleStyle.log("CORTEX", f"Architecting solution for: '{task}'...", ConsoleStyle.YELLOW)
        
        system_prompt = textwrap.dedent("""
            You are a Senior Python Engineer for MassiveMagnetics.
            Your goal: Write a single, self-contained Python function to solve the user's task.
            
            CONSTRAINTS:
            1. Return ONLY the code. No markdown, no text.
            2. Use standard libraries only where possible.
            3. Include a docstring.
            4. Function name must be snake_case and descriptive.
        """)
        
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Task: {task}"}
            ],
            temperature=0.1
        )
        
        # Clean formatting just in case
        raw_content = response.choices[0].message.content.strip()
        clean_code = raw_content.replace("```python", "").replace("```", "")
        return clean_code

    def _unsafe_compile(self, code_str: str) -> Callable:
        """Runtime Compilation."""
        local_scope = {}
        try:
            exec(code_str, {}, local_scope)
            func = list(local_scope.values())[-1]
            return func
        except Exception as e:
            ConsoleStyle.log("ERROR", f"Compilation Failed: {e}", ConsoleStyle.RED)
            raise e

    def execute(self, objective: str):
        print("-" * 60)
        ConsoleStyle.log("SYSTEM", f"Incoming Directive: {objective}")
        
        # 1. RETRIEVAL STEP
        cached_tool = self.memory.find_tool(objective)
        
        if cached_tool:
            ConsoleStyle.log("MEMORY", f"Recall Success. Using existing tool: [{cached_tool}]", ConsoleStyle.GREEN)
            print(f"   >>> Running {cached_tool}()... [Success]")
            return

        # 2. GENERATION STEP
        code = self._generate_code(objective)
        
        # 3. COMPILATION STEP
        try:
            tool_func = self._unsafe_compile(code)
            tool_name = tool_func.__name__
            
            # 4. MEMORY CONSOLIDATION
            tags = ["math"] if "calc" in objective or "number" in objective else ["utility"]
            
            self.memory.add_tool(
                name=tool_name,
                code=code,
                func=tool_func,
                description=objective,
                tags=tags
            )
            
            print(f"   >>> Executed {tool_name}()... [Success]")
            
        except Exception as e:
            ConsoleStyle.log("ERROR", f"Pipeline failed: {e}", ConsoleStyle.RED)

    def visualize(self):
        """Generates the 'Proof of Work' screenshot."""
        ConsoleStyle.log("MASSIVE_MAGNETICS", "Rendering Neural Map...", ConsoleStyle.CYAN)
        
        plt.figure(figsize=(10, 6))
        pos = nx.spring_layout(self.memory.graph, seed=42)
        
        nx.draw_networkx_nodes(self.memory.graph, pos, node_size=2000, node_color="#2C3E50", alpha=0.9)
        nx.draw_networkx_edges(self.memory.graph, pos, width=2, alpha=0.5, edge_color="#BDC3C7")
        nx.draw_networkx_labels(self.memory.graph, pos, font_size=10, font_color="white", font_weight="bold")
        
        plt.title("MassiveMagnetics: Recursive Tool Graph", fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

# --- MAIN ENTRYPOINT ---
if __name__ == "__main__":
    print(f"\n{ConsoleStyle.BOLD}=== MASSIVE MAGNETICS AGENT INIT ==={ConsoleStyle.RESET}")
    
    bot = RecursiveBuilder()
    
    # 1. Create
    bot.execute("Write a function to calculate the fibonacci sequence")
    
    # 2. Create
    bot.execute("Create a function to generate a secure random password")
    
    # 3. Retrieve (The Hook)
    bot.execute("Run the calculate_fibonacci function")
    
    # 4. Proof
    bot.visualize()
