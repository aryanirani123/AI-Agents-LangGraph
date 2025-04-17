"""

This Python script implements a web search and summarization agent using LangGraph, 
DuckDuckGo for web search, and Google Gemini for summarization.

Key Features:
- Takes a user's natural language query as input.
- Utilizes DuckDuckGo to search the web for relevant information.
- Employs Google Gemini to generate a concise summary of the search results,
  specifically tailored to answer the user's query.
- Implements a LangGraph state graph to manage the workflow, error handling, 
  and transitions between search and summarization steps.

Dependencies:
- google-generativeai: For interacting with Google Gemini models.
- langchain-community: For accessing the DuckDuckGo search tool.
- langgraph: For building the stateful agent.
- typing: For type hints.
- os: For accessing environment variables (API key).

Environment Variable:
- GOOGLE_API_KEY:  Must be set with your Google Gemini API key.


"""



import os
import google.generativeai as genai
from typing import TypedDict, Optional, List
from langgraph.graph import StateGraph, END

# Import the DuckDuckGo search tool from LangChain community
from langchain_community.tools import DuckDuckGoSearchRun

# --- Environment Variable for API Key (Recommended) ---

try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    print("🛑 Error: GOOGLE_API_KEY environment variable not set.")
    print("Please set the environment variable before running the script.")
    exit() # Exit if the key isn't set

# --- 1. Define the State ---

class SearchSummarizeState(TypedDict):
    query: str                 # The initial natural language query (e.g., "add 5 and 10")
    search_results : Optional[str]
    summary: Optional[str]
    error: Optional[str]

search_tool = DuckDuckGoSearchRun()

# --- 1. Define the Tools ---

def search_web(state: SearchSummarizeState) -> SearchSummarizeState:
    """
    Uses DuckDuckGo to search the web based on the query in the state.
    """
    print("\n--- Node: search_web ---")
    query = state.get("query")
    if not query:
         print("Error: No query provided for search.")
         return {**state, "error": "No query provided for search.", "search_results": None}

    print(f"Searching the web for: '{query}'")
    try:
        # Use the initialized search tool
        results = search_tool.run(query)
        print(f"Search results obtained (first 100 chars): {results[:100]}...")
        return {**state, "search_results": results, "error": None}
    except Exception as e:
            error_message = f"Web search failed for query '{query}': {e}"
            print(f"Error: {error_message}")
            return {**state, "search_results": None, "error": error_message}


# --- Node 3b: Summarize Search Results ---
def summarize_results(state: SearchSummarizeState) -> SearchSummarizeState:
    """
    Uses Google Gemini to summarize the search results based on the original query.
    """
    print("--- Node: summarize_results ---")
    # Check if an error occurred in the previous step
    if state.get("error"):
        print(f"Skipping summarization due to previous error: {state['error']}")
        return state # Pass the existing error along

    query = state.get("query")
    search_results = state.get("search_results")

    if not search_results:
        print("Error: No search results available for summarization.")
        # Ensure error state is consistent
        current_error = state.get("error")
        return {**state, "error": current_error if current_error else "No search results found to summarize."}

    print(f"Summarizing results for query: '{query}'")

    # Configure the Gemini model
    model = genai.GenerativeModel('gemini-1.5-flash') # Or 'gemini-pro'

    # Prompt for summarization
    prompt = f"""Based on the following search results, please provide a concise summary that directly answers the query: "{query}"

Search Results:
---
{search_results}
---

Concise Summary:"""

    try:
        response = model.generate_content(prompt)
        if response.parts:
             summary_text = response.text.strip()
             print(f"LLM generated summary: {summary_text}")
             return {**state, "summary": summary_text, "error": None}
        else:
             # Handle cases where the response might be blocked or empty
             error_message = f"LLM did not return a valid summary for query '{query}'. Response: {response}"
             print(f"Error: {error_message}")
             return {**state, "summary": None, "error": error_message}

    except Exception as e:
        error_message = f"LLM summarization failed for query '{query}': {e}"
        print(f"Error: {error_message}")
        return {**state, "summary": None, "error": error_message}
    



# --- 4. Define the Graph ---
workflow = StateGraph(SearchSummarizeState)

# Add the nodes
workflow.add_node("searcher", search_web)
workflow.add_node("summarizer", summarize_results)

# --- 5. Define the Edges ---
# Set the entry point
workflow.set_entry_point("searcher")

# Define the flow: searcher -> summarizer -> END
workflow.add_edge("searcher", "summarizer")
workflow.add_edge("summarizer", END)

# --- 6. Compile the Graph ---
print("\nCompiling the LangGraph agent...")
app = workflow.compile()
print("Agent compiled successfully.")

# --- 7. Interactive User Input Loop ---
print("\n--- Web Search & Summarizer Agent ---")
print("Enter your search query (e.g., 'What is LangGraph?', 'Latest news on AI').")
print("Type 'quit' or 'exit' to stop.")


while True:
    try:
        user_query = input("\n➡️ Query: ").strip()

        if user_query.lower() in ['quit', 'exit']:
            print("\nExiting agent. Goodbye! 👋")
            break

        if not user_query:
            print("Please enter a query.")
            continue

        # Prepare the initial state for the LangGraph app
        initial_state = {"query": user_query}

        print("🧠 Processing...")
        # Invoke the LangGraph agent
        final_state = app.invoke(initial_state)

        # Display the results
        print("\n--- Result ---")
        print(f"  🗣️ Initial Query:    {final_state.get('query')}")
        # Optionally display raw search results (can be long)
        # search_res = final_state.get('search_results', 'N/A')
        # print(f"  🔍 Search Snippets: {search_res[:200] + '...' if len(search_res) > 200 else search_res}") # Show preview
        if final_state.get('error'):
            print(f"  ❌ Error:            {final_state.get('error')}")
        else:
            print(f"  ✅ Summary:          {final_state.get('summary', 'N/A')}") # Show N/A if no summary

    except KeyboardInterrupt:
        print("\nExiting agent. Goodbye! 👋")
        break
    except Exception as e:
        print(f"\n🛑 An unexpected error occurred: {e}")
        print("Please try again or type 'quit' to exit.")
