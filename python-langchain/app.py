from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Load environment variables from .env file
load_dotenv()


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
    
    # Create a test query
    query = "What is 25 * 4 + 10?"
    print(f"\n📝 Query: {query}")
    
    # Invoke the LLM with the query (will attempt to answer without tools)
    response = client.invoke([HumanMessage(content=query)])
    print(f"🤖 Response: {response.content}")


if __name__ == "__main__":
    main()
