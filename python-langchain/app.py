from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain_core.tools import Tool, tool
from datetime import datetime

# Load environment variables from .env file
load_dotenv()


def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression safely and return the result as a string.
    
    For demo purposes, uses eval() with basic input validation.
    In production, use a proper expression parser like SymPy or NumPy.
    
    Args:
        expression: A string containing a mathematical expression (e.g., "25 * 4 + 10")
    
    Returns:
        A string containing the result or an error message.
    """
    try:
        # Basic safety: only allow alphanumeric, operators, parentheses, decimals
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return "❌ Error: Invalid characters in expression"
        
        # Evaluate the expression
        result = eval(expression)
        return str(result)
    except ZeroDivisionError:
        return "❌ Error: Division by zero"
    except SyntaxError:
        return "❌ Error: Invalid syntax in expression"
    except Exception as e:
        return f"❌ Error: {type(e).__name__}: {str(e)}"


def main():
    """Main entry point for the LangChain AI Agent application."""
    import os
    
    print("🚀 Starting LangChain AI Agent Application...")
    
    # Check if GITHUB_TOKEN is set
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        print("\n❌ Error: GITHUB_TOKEN not found in environment variables!")
        print("\n📋 To fix this, follow these steps:")
        print("   1. Create a .env file in the project root directory")
        print("   2. Add your GitHub token: GITHUB_TOKEN=your_token_here")
        print("   3. Get a token from: https://github.com/settings/tokens")
        print("\n💡 Tip: The .env file should NOT be committed to version control.\n")
        return
    
    print("✅ GitHub token loaded successfully!")
    
    # Create ChatOpenAI instance
    client = ChatOpenAI(
        model="openai/gpt-4o",
        temperature=0,
        base_url="https://models.github.ai/inference",
        api_key=github_token
    )
    print("🤖 ChatOpenAI client initialized successfully!")
    
    # Create a tools list with the calculator tool
    tools = [
        Tool(
            name="Calculator",
            func=calculator,
            description="Use this tool to evaluate mathematical expressions and perform calculations. "
                       "Provide mathematical expressions like '25 * 4 + 10' or '100 / 5 - 2'. "
                       "The tool supports basic arithmetic operations: +, -, *, /, and parentheses. "
                       "Use this whenever you need to perform precise calculations or when the user asks for math."
        )
    ]
    print("🛠️ Tools loaded successfully!")
    
    # Bind tools to the LLM
    llm_with_tools = client.bind_tools(tools)
    print("⚙️ Tools bound to LLM successfully!")
    
    # Create a test query
    query = "What is 25 * 4 + 10?"
    print(f"\n📝 Query: {query}")
    
    # Simple agent loop: invoke LLM with tools
    try:
        messages = [HumanMessage(content=query)]
        response = llm_with_tools.invoke(messages)
        
        # Check if the LLM wants to use a tool
        if response.tool_calls:
            print(f"🔧 LLM called tool: {response.tool_calls[0]['name']}")
            tool_name = response.tool_calls[0]["name"]
            tool_input = response.tool_calls[0]["args"]
            
            # Find and execute the tool
            for tool in tools:
                if tool.name == tool_name:
                    # Handle both dictionary and positional argument formats
                    if isinstance(tool_input, dict) and tool_input:
                        # If args is a dict, use it; otherwise get first value
                        if "expression" in tool_input:
                            tool_result = tool.func(tool_input["expression"])
                        else:
                            # Get the first value from the dict
                            tool_result = tool.func(list(tool_input.values())[0])
                    else:
                        tool_result = tool.func(tool_input)
                    print(f"🤖 Tool result: {tool_result}")
                    break
        else:
            print(f"🤖 Response: {response.content}")
    except Exception as e:
        print(f"❌ Error during agent execution: {type(e).__name__}: {str(e)}")


if __name__ == "__main__":
    main()
