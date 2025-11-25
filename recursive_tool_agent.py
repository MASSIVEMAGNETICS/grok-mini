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
import re
import textwrap
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, Callable, Optional, List
from openai import OpenAI

# --- CONFIGURATION ---
# PRO TIP: Export this in your terminal: export OPENAI_API_KEY="sk-..."
API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_ID = "gpt-4o"

# Client will be initialized on first use
client = None

def _get_client():
    """Lazy initialization of OpenAI client."""
    global client
    if client is None:
        if not API_KEY:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required. "
                "Please set it before running: export OPENAI_API_KEY='sk-...'"
            )
        client = OpenAI(api_key=API_KEY)
    return client

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
        # 1. Direct Name Match (exact)
        for name in self.tools:
            if name == query:
                return name
        
        # 2. Substring Match (bidirectional fuzzy)
        clean_query = query.lower().replace(" ", "_")
        for name in self.tools:
            name_lower = name.lower()
            # Check if tool name is in query OR query is in tool name
            if name_lower in clean_query or clean_query in name_lower:
                return name
            # Check if any word from query matches tool name
            for word in clean_query.split('_'):
                if len(word) > 3 and word in name_lower:  # Skip short words
                    return name
        return None

class RecursiveBuilder:
    """
    The Agent: Capable of self-modification via tool creation.
    """
    def __init__(self):
        self.memory = ToolGraphMemory()

    def _classify_tags(self, objective: str) -> List[str]:
        """Classify tool into semantic categories based on objective."""
        objective_lower = objective.lower()
        
        tags = []
        
        # Math/calculation
        if any(word in objective_lower for word in ["calc", "number", "math", "fibonacci", "sum", "multiply", "add"]):
            tags.append("math")
        
        # String/text processing
        if any(word in objective_lower for word in ["string", "text", "password", "format", "parse"]):
            tags.append("text")
        
        # Data structures
        if any(word in objective_lower for word in ["list", "dict", "array", "sort", "search"]):
            tags.append("data_structure")
        
        # File/IO operations
        if any(word in objective_lower for word in ["file", "read", "write", "save", "load"]):
            tags.append("io")
        
        # Default to utility if no specific category
        if not tags:
            tags.append("utility")
        
        return tags

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
        
        response = _get_client().chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Task: {task}"}
            ],
            temperature=0.1
        )
        
        # Clean formatting (handle various markdown code block formats)
        raw_content = response.choices[0].message.content.strip()
        
        # Remove markdown code blocks (python, py, or no language specified)
        import re
        # Match ```python, ```py, or just ```
        code_block_pattern = r'^```(?:python|py)?\s*\n(.*?)\n```$'
        match = re.search(code_block_pattern, raw_content, re.DOTALL)
        
        if match:
            clean_code = match.group(1)
        else:
            # Fallback to simple replacement for inline cases
            clean_code = raw_content.replace("```python", "").replace("```py", "").replace("```", "")
        
        return clean_code.strip()

    def _unsafe_compile(self, code_str: str) -> Callable:
        """
        Runtime Compilation.
        
        WARNING: This uses exec() to compile LLM-generated code, which poses security risks.
        Only use this in trusted environments with controlled inputs. Do NOT expose this
        to untrusted users or run in production without proper sandboxing.
        
        For production use, consider:
        - Running in a sandboxed environment (Docker, VM, restricted Python environment)
        - Code review/validation before execution
        - Static analysis of generated code
        - Input validation and sanitization
        """
        local_scope = {}
        try:
            exec(code_str, {}, local_scope)
            
            # Extract callable functions from the scope
            functions = [v for v in local_scope.values() if callable(v)]
            
            if len(functions) == 0:
                raise ValueError("Generated code did not produce any callable function")
            elif len(functions) > 1:
                # If multiple functions, prefer the last one (main function)
                # or the one with the longest name (typically the main implementation)
                func = max(functions, key=lambda f: len(f.__name__))
            else:
                func = functions[0]
            
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
            # Actually execute the cached tool
            try:
                result = self.memory.tools[cached_tool]()
                print(f"   >>> Executed {cached_tool}() -> {result}")
            except TypeError:
                # Function may require arguments
                print(f"   >>> Found {cached_tool}() [Tool available for use]")
            return

        # 2. GENERATION STEP
        code = self._generate_code(objective)
        
        # 3. COMPILATION STEP
        try:
            tool_func = self._unsafe_compile(code)
            tool_name = tool_func.__name__
            
            # 4. MEMORY CONSOLIDATION
            tags = self._classify_tags(objective)
            
            self.memory.add_tool(
                name=tool_name,
                code=code,
                func=tool_func,
                description=objective,
                tags=tags
            )
            
            print(f"   >>> Generated and stored {tool_name}() [Success]")
            
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
