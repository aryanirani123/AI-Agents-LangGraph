"""
This Python script implements a natural language calculator using LangGraph and Google Gemini.

Key Features:
- Accepts natural language mathematical queries (e.g., "add 5 and 10").
- Employs Google Gemini to translate the natural language query into a Python-evaluable
  mathematical expression.
- Evaluates the generated expression using Python's `eval()` function (with a security warning).
- Implements a LangGraph state graph to manage the workflow, error handling,
  and transitions between translation and evaluation steps.


Dependencies:
- google-generativeai: For interacting with Google Gemini models.
- langgraph: For building the stateful agent.
- typing: For type hints.
- os: For accessing environment variables (API key).

Environment Variable:
- GOOGLE_API_KEY: Must be set with your Google Gemini API key.


"""

import os
import google.generativeai as genai
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

# --- Environment Variable for API Key (Recommended) ---
# Make sure you have set the GOOGLE_API_KEY environment variable
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    print("🛑 Error: GOOGLE_API_KEY environment variable not set.")
    print("Please set the environment variable before running the script.")
    exit() # Exit if the key isn't set

# --- 1. Define the Updated State ---
# Add a 'query' field for the initial natural language input.
# Make 'expression' optional as it will be generated by the LLM.
class CalculatorState(TypedDict):
    query: str                 # The initial natural language query (e.g., "add 5 and 10")
    expression: Optional[str]  # The mathematical expression generated by the LLM (e.g., "5 + 10")
    result: Optional[float]    # The result of the calculation
    error: Optional[str]       # To store any potential error messages

# --- 2. Define the Nodes ---

# --- Node 2a: Translate Query to Expression (NEW) ---
def translate_to_expression(state: CalculatorState) -> CalculatorState:
    """
    Uses Google Gemini to translate the natural language query
    into a Python-evaluable mathematical expression.
    """
    print("--- Node: translate_to_expression ---")
    query = state.get("query")
    if not query:
        print("Error: No query provided.")
        return {**state, "error": "No query provided.", "expression": None}

    print(f"Translating query: '{query}'")

    # Configure the Gemini model
    # You might want to choose a specific model version, e.g., 'gemini-1.5-flash'
    model = genai.GenerativeModel('gemini-1.5-flash')

    # Simple prompt engineering - instruct the LLM clearly
    prompt = f"""Translate the following user query into a simple, standard mathematical expression suitable for Python's eval() function.
Only output the mathematical expression itself, nothing else.

Examples:
Query: add 5 and 10
Expression: 5 + 10

Query: what is 100 divided by 4?
Expression: 100 / 4

Query: calculate 3 times the sum of 2 and 8
Expression: 3 * (2 + 8)

Query: 50 minus 15
Expression: 50 - 15

Now, translate this query:
Query: {query}
Expression:"""

    try:
        response = model.generate_content(prompt)
        # Basic check if the response has text
        if response.parts:
             generated_expression = response.text.strip()
             print(f"LLM generated expression: '{generated_expression}'")
             return {**state, "expression": generated_expression, "error": None}
        else:
             # Handle cases where the response might be blocked or empty
             error_message = f"LLM did not return a valid expression for query '{query}'. Response: {response}"
             print(f"Error: {error_message}")
             return {**state, "expression": None, "error": error_message}

    except Exception as e:
        error_message = f"LLM call failed for query '{query}': {e}"
        print(f"Error: {error_message}")
        return {**state, "expression": None, "error": error_message}


# --- Node 2b: Evaluate Expression (Slightly Modified) ---
def evaluate_expression(state: CalculatorState) -> CalculatorState:
    """
    Evaluates the mathematical expression generated by the LLM.
    Updates the 'result' or 'error' field in the state.
    """
    print("--- Node: evaluate_expression ---")
    # Check if an error occurred in the previous step
    if state.get("error"):
        print(f"Skipping evaluation due to previous error: {state['error']}")
        return state # Pass the existing error along

    expression = state.get("expression")
    if not expression:
        print("Error: No expression provided (likely LLM translation failed).")
        # Ensure error state is consistent if expression is missing
        current_error = state.get("error")
        return {**state, "error": current_error if current_error else "No expression provided for evaluation."}

    try:
        # WARNING: eval() is generally unsafe with untrusted input,
        # even if generated by an LLM. Use with caution.
        print(f"Evaluating: {expression}")
        calculation_result = eval(expression)
        print(f"Result: {calculation_result}")
        # Return the updated state dictionary
        return {**state, "result": float(calculation_result), "error": None}
    except Exception as e:
        error_message = f"Failed to evaluate expression '{expression}': {e}"
        print(f"Error: {error_message}")
        # Return the updated state dictionary with the error
        return {**state, "result": None, "error": error_message}

# --- 3. Define the Graph ---
workflow = StateGraph(CalculatorState)

# Add the nodes to the graph
workflow.add_node("translator", translate_to_expression)
workflow.add_node("calculator", evaluate_expression)

# --- 4. Define the Edges ---
# Set the entry point to the new translator node
workflow.set_entry_point("translator")

# Define the flow: translator -> calculator -> END
workflow.add_edge("translator", "calculator")
workflow.add_edge("calculator", END)

# --- 5. Compile the Graph ---
app = workflow.compile()

# --- 6. Run the Agent ---

# Example 1: Simple natural language addition
print("\n--- Running Example 1 ---")
inputs1 = {"query": "what is 5 plus 12?"}
final_state1 = app.invoke(inputs1)
print("\n--- Final State 1 ---")
print(final_state1)
# Expected Output (approx): {'query': 'what is 5 plus 12?', 'expression': '5 + 12', 'result': 17.0, 'error': None}

# Example 2: More complex natural language expression
print("\n--- Running Example 2 ---")
inputs2 = {"query": "calculate 10 times the result of 8 minus 3"}
final_state2 = app.invoke(inputs2)
print("\n--- Final State 2 ---")
print(final_state2)
# Expected Output (approx): {'query': 'calculate 10 times the result of 8 minus 3', 'expression': '10 * (8 - 3)', 'result': 50.0, 'error': None}

# Example 3: Query leading to division by zero after translation
print("\n--- Running Example 3 ---")
inputs3 = {"query": "divide 100 by zero"}
final_state3 = app.invoke(inputs3)
print("\n--- Final State 3 ---")
print(final_state3)
# Expected Output (approx): {'query': 'divide 100 by zero', 'expression': '100 / 0', 'result': None, 'error': "Failed to evaluate expression '100 / 0': division by zero"}

# Example 4: Query that might confuse the LLM or lead to invalid expression
print("\n--- Running Example 4 ---")
inputs4 = {"query": "what is the square root of apple?"} # Non-mathematical
final_state4 = app.invoke(inputs4)
print("\n--- Final State 4 ---")
print(final_state4)
# Expected Output (approx): Might result in an error either at the LLM stage or eval stage, e.g.,
# {'query': 'what is the square root of apple?', 'expression': 'sqrt(apple)', 'result': None, 'error': "Failed to evaluate expression 'sqrt(apple)': name 'sqrt' is not defined"}
# OR potentially an error from the 'translator' node if the LLM fails badly.
