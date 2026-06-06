import logging
import os
import time
from datetime import datetime
from pathlib import Path

import openai
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables from .env file
load_dotenv()

# ─── Logging Configuration ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ai_agent")


# ─── Custom Exceptions ──────────────────────────────────────────────────────
class AgentError(Exception):
    """Base exception for AI agent errors."""
    pass


class RateLimitError(AgentError):
    """Raise when API rate limit is hit (HTTP 429)."""
    pass


class InputValidationError(AgentError):
    """Raise when input validation fails."""
    pass


# ─── Performance Tracking ───────────────────────────────────────────────────
class PerformanceMonitor:
    """Track and log performance metrics for AI queries and tool calls."""

    def __init__(self, slow_query_threshold: float = 10.0):
        self.slow_query_threshold = slow_query_threshold
        self.metrics: list[dict[str, object]] = []

    def record_query(self, query: str, duration: float, tool_calls: int = 0) -> None:
        """Record a query's performance metrics."""
        entry = {
            "query": query,
            "duration_s": round(duration, 3),
            "tool_calls": tool_calls,
            "timestamp": datetime.now().isoformat(),
            "slow": duration > self.slow_query_threshold,
        }
        self.metrics.append(entry)
        logger.info(
            "Query completed in %.3fs with %d tool call(s): %s",
            duration,
            tool_calls,
            query[:80],
        )
        if entry["slow"]:
            logger.warning(
                "Slow query detected (%.3fs > %.1fs threshold): %s",
                duration,
                self.slow_query_threshold,
                query[:120],
            )

    def summary(self) -> str:
        """Return a summary of all recorded metrics."""
        if not self.metrics:
            return "No queries recorded."
        total = len(self.metrics)
        avg_time = sum(m["duration_s"] for m in self.metrics) / total
        slow_count = sum(1 for m in self.metrics if m["slow"])
        total_tool_calls = sum(m["tool_calls"] for m in self.metrics)
        lines = [
            f"  Total queries: {total}",
            f"  Average response time: {avg_time:.3f}s",
            f"  Slow queries (>{self.slow_query_threshold}s): {slow_count}",
            f"  Total tool calls: {total_tool_calls}",
        ]
        return "\n".join(lines)


# ─── Input Validation ───────────────────────────────────────────────────────
MAX_QUERY_LENGTH = 500
ALLOWED_QUERY_CHARS = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789 .,!?;:'\"-()[]{}/@#$%^&*+=<>~\n\t"
)


def validate_query(query: str) -> str:
    """Validate and sanitize a user query before sending to the AI.

    Args:
        query: The raw user query string.

    Returns:
        The sanitized query string.

    Raises:
        InputValidationError: If the query fails validation.
    """
    if not query or not query.strip():
        raise InputValidationError("Query cannot be empty.")

    sanitized = query.strip()

    if len(sanitized) > MAX_QUERY_LENGTH:
        raise InputValidationError(
            f"Query exceeds maximum length of {MAX_QUERY_LENGTH} characters "
            f"(got {len(sanitized)})."
        )

    # Check for suspicious characters
    invalid_chars = set(sanitized) - ALLOWED_QUERY_CHARS
    if invalid_chars:
        logger.warning("Query contains unusual characters: %s", invalid_chars)

    return sanitized


# ─── Tool Functions ──────────────────────────────────────────────────────────

def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely and return the result.

    Use a restricted set of allowed characters and a restricted eval namespace
    for improved security. Support basic arithmetic: +, -, *, /, parentheses,
    and decimals.

    Args:
        expression: A string containing a mathematical expression (e.g., "25 * 4 + 10").

    Returns:
        A string containing the result or an error message.
    """
    try:
        # Validate input characters
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"

        # Additional safety: check for dangerous patterns
        dangerous_patterns = ["__", "import", "exec", "eval", "open", "file"]
        expression_lower = expression.lower()
        for pattern in dangerous_patterns:
            if pattern in expression_lower:
                return f"Error: Dangerous pattern '{pattern}' detected"

        # Evaluate the expression in a restricted namespace
        safe_globals = {"__builtins__": {}}
        result = eval(expression, safe_globals, {})  # noqa: S307
        return str(result)
    except ZeroDivisionError:
        return "Error: Division by zero"
    except SyntaxError:
        return "Error: Invalid syntax in expression"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


def get_current_time(input_str: str) -> str:
    """Return the current date and time.

    Args:
        input_str: Required by Tool interface (unused for this function).

    Returns:
        A string containing the current date and time in YYYY-MM-DD HH:MM:SS format.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def reverse_string(input_string: str) -> str:
    """Reverse a string and return the result.

    Args:
        input_string: A string to reverse.

    Returns:
        The reversed string.
    """
    return input_string[::-1]


def get_weather_by_date(date_str: str) -> str:
    """Return mock weather information for a given date.

    If the date matches today, return sunny weather; otherwise return rainy weather.

    Args:
        date_str: A date string in the format "YYYY-MM-DD".

    Returns:
        A string containing the weather information for the given date.
    """
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        if date_str == today:
            return "Sunny, 72°F"
        else:
            return "Rainy, 55°F"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


def get_weather_by_city(city: str) -> str:
    """Return mock weather information for a given city.

    For cities not in the mock data, return a default weather response.

    Args:
        city: The name of the city (e.g., "New York", "London").

    Returns:
        A string containing the simulated weather information for the city.
    """
    try:
        # Mock weather data for various cities
        weather_data = {
            "new york": "Partly Cloudy, 75°F",
            "london": "Overcast, 62°F",
            "paris": "Sunny, 71°F",
            "tokyo": "Rainy, 68°F",
            "sydney": "Clear, 78°F",
            "los angeles": "Sunny, 82°F",
            "chicago": "Windy, 65°F",
            "miami": "Hot and Humid, 88°F",
            "seattle": "Rainy, 58°F",
            "denver": "Clear, 70°F",
            "boston": "Cloudy, 60°F",
            "san francisco": "Foggy, 64°F",
            "louisville": "Partly Sunny, 74°F",
            "atlanta": "Warm, 80°F",
            "dallas": "Hot, 92°F",
        }
        city_lower = city.lower().strip()
        if city_lower in weather_data:
            return f"Weather in {city.title()}: {weather_data[city_lower]}"
        else:
            return f"Weather in {city.title()}: Partly Cloudy, 70°F (default forecast)"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


def file_tool(operation: str) -> str:
    """Read from or write to a text file in the project directory.

    Supports two operations:
    - "read:<filename>" — read the contents of the specified file
    - "write:<filename>:<content>" — write content to the specified file

    Args:
        operation: A string in the format "read:<filename>" or
                    "write:<filename>:<content>".

    Returns:
        A string containing the file contents (for read) or a confirmation
        message (for write), or an error message.
    """
    try:
        parts = operation.split(":", 2)
        action = parts[0].lower().strip()

        if action == "read" and len(parts) >= 2:
            filename = parts[1].strip()
            # Security: only allow .txt files in the current directory
            if "/" in filename or "\\" in filename or ".." in filename:
                return "Error: Only files in the current directory are allowed"
            if not filename.endswith(".txt"):
                return "Error: Only .txt files are supported"
            file_path = Path(filename)
            if not file_path.exists():
                return f"Error: File '{filename}' not found"
            content = file_path.read_text(encoding="utf-8")
            logger.info("File read: %s (%d chars)", filename, len(content))
            return f"Contents of {filename}:\n{content}"

        elif action == "write" and len(parts) >= 3:
            filename = parts[1].strip()
            content = parts[2]
            # Security: only allow .txt files in the current directory
            if "/" in filename or "\\" in filename or ".." in filename:
                return "Error: Only files in the current directory are allowed"
            if not filename.endswith(".txt"):
                return "Error: Only .txt files are supported"
            file_path = Path(filename)
            file_path.write_text(content, encoding="utf-8")
            logger.info("File written: %s (%d chars)", filename, len(content))
            return f"Successfully wrote {len(content)} characters to {filename}"

        else:
            return (
                "Error: Invalid operation format. "
                "Use 'read:<filename>' or 'write:<filename>:<content>'"
            )
    except Exception as e:
        logger.error("File tool error: %s", e)
        return f"Error: {type(e).__name__}: {e}"


def web_search(query: str) -> str:
    """Simulate a web search and return mock search results.

    For unknown topics, return a generic response.

    Args:
        query: The search query string.

    Returns:
        A string containing simulated search results.
    """
    try:
        # Mock search results database
        search_results = {
            "python": (
                "Python Programming Language\n"
                "1. Python Official Site - https://www.python.org\n"
                "2. Python Documentation - https://docs.python.org\n"
                "3. Python Tutorial - W3Schools\n"
                "Python is a popular programming language known for its "
                "simplicity and readability."
            ),
            "langchain": (
                "LangChain Framework\n"
                "1. LangChain Documentation - https://docs.langchain.com\n"
                "2. LangChain GitHub - https://github.com/langchain-ai\n"
                "3. LangChain Tutorials - Quick Start Guide\n"
                "LangChain is a framework for building applications with LLMs."
            ),
            "ai": (
                "Artificial Intelligence\n"
                "1. AI Overview - Wikipedia\n"
                "2. OpenAI - https://openai.com\n"
                "3. AI News - Latest developments in AI\n"
                "AI is the simulation of human intelligence by machines."
            ),
            "weather": (
                "Weather Information\n"
                "1. National Weather Service - https://weather.gov\n"
                "2. Weather.com - The Weather Channel\n"
                "3. AccuWeather - https://accuweather.com\n"
                "Current weather data available for locations worldwide."
            ),
            "github": (
                "GitHub\n"
                "1. GitHub - https://github.com\n"
                "2. GitHub Docs - https://docs.github.com\n"
                "3. GitHub Models - AI model marketplace\n"
                "GitHub is a platform for version control and collaboration."
            ),
        }

        query_lower = query.lower().strip()
        # Check for keyword matches
        for keyword, results in search_results.items():
            if keyword in query_lower:
                logger.info("Web search for '%s' matched keyword '%s'", query, keyword)
                return results

        # Default response for unknown queries
        logger.info("Web search for '%s' returned default results", query)
        return (
            f"Search results for: '{query}'\n"
            "1. No exact matches found.\n"
            "2. Try refining your search terms.\n"
            "3. Related topics may be available.\n"
            f"Simulated search completed for: {query}"
        )
    except Exception as e:
        logger.error("Web search error: %s", e)
        return f"Error: {type(e).__name__}: {e}"


# ─── Main Application ───────────────────────────────────────────────────────

def main():
    """Run the LangChain AI Agent application demo."""
    logger.info("Python LangChain Agent Starting...")
    print("Python LangChain Agent Starting...")

    # Check if GITHUB_TOKEN is set
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        logger.error("GITHUB_TOKEN not found in environment variables")
        print("\nError: GITHUB_TOKEN not found in environment variables!")
        print("\nTo fix this, follow these steps:")
        print("   1. Create a .env file in the project root directory")
        print("   2. Add your GitHub token: GITHUB_TOKEN=your_token_here")
        print("   3. Get a token from: https://github.com/settings/tokens")
        print("\nTip: The .env file should NOT be committed to version control.\n")
        return

    logger.info("GitHub token loaded successfully")
    print("GitHub token loaded successfully!")

    # Create ChatOpenAI instance with retry configuration
    llm = ChatOpenAI(
        model="openai/gpt-4o",
        temperature=0,
        base_url="https://models.github.ai/inference",
        api_key=github_token,
        max_retries=3,
    )
    logger.info("ChatOpenAI client initialized")
    print("ChatOpenAI client initialized successfully!")

    # Create a tools list
    tools = [
        Tool(
            name="Calculator",
            func=calculator,
            description=(
                "Use this tool to evaluate mathematical expressions and perform "
                "calculations. Provide mathematical expressions like '25 * 4 + 10' "
                "or '100 / 5 - 2'. The tool supports basic arithmetic operations: "
                "+, -, *, /, and parentheses. Use this whenever you need to "
                "perform precise calculations or when the user asks for math."
            ),
        ),
        Tool(
            name="get_current_time",
            func=get_current_time,
            description=(
                "Use this tool to get the current date and time. Call this "
                "whenever the user asks for the time, current date, or any "
                "time-related information. Returns the time in "
                "YYYY-MM-DD HH:MM:SS format."
            ),
        ),
        Tool(
            name="reverse_string",
            func=reverse_string,
            description="Reverses a string. Input should be a single string.",
        ),
        Tool(
            name="get_weather",
            func=get_weather_by_date,
            description=(
                "Use this tool to get weather information for a given date. "
                "Call this whenever the user asks about the weather. "
                "Input should be a date string in the format YYYY-MM-DD. "
                "Returns weather information including conditions and temperature."
            ),
        ),
        Tool(
            name="get_weather_by_city",
            func=get_weather_by_city,
            description=(
                "Use this tool to get weather information for a given city. "
                "Call this whenever the user asks about the weather in a "
                "specific city. Input should be a city name like 'New York' "
                "or 'London'. Returns simulated weather information for the city."
            ),
        ),
        Tool(
            name="file_tool",
            func=file_tool,
            description=(
                "Use this tool to read from or write to text files. "
                "For reading, use format: 'read:<filename>'. "
                "For writing, use format: 'write:<filename>:<content>'. "
                "Only .txt files in the current directory are supported. "
                "Examples: 'read:notes.txt' or 'write:notes.txt:Hello World'"
            ),
        ),
        Tool(
            name="web_search",
            func=web_search,
            description=(
                "Use this tool to search the web for information. "
                "Call this whenever the user asks about a topic that requires "
                "looking up information online. Input should be a search query "
                "string. Returns simulated search results."
            ),
        ),
    ]
    logger.info("Tools loaded: %s", [t.name for t in tools])
    print(f"Tools loaded successfully! ({len(tools)} tools)")

    # Create agent with system message and conversation memory.
    # MemorySaver provides in-memory checkpointing so the agent remembers
    # previous messages within the same thread, enabling multi-turn conversations.
    system_prompt = (
        "You are a professional and succinct AI assistant. "
        "When asked a question, use the available tools to provide accurate answers. "
        "Be concise in your responses. You have access to tools for calculations, "
        "time, string manipulation, weather lookups, file operations, and web search."
    )

    memory = MemorySaver()
    logger.info("Conversation memory (MemorySaver) initialized")
    print("Conversation memory initialized!")

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=memory,
        debug=False,
    )
    logger.info("Agent created successfully with conversation memory")
    print("Agent created successfully!")

    # Initialize performance monitor
    perf = PerformanceMonitor(slow_query_threshold=10.0)

    # Use a consistent thread_id so the agent remembers previous messages
    # across queries in this session.
    config = {"configurable": {"thread_id": "demo-session-1"}}
    logger.info("Using conversation thread: %s", config["configurable"]["thread_id"])

    # Test queries covering all tools
    queries = [
        "What time is it right now?",
        "What is 25 * 4 + 10?",
        "Reverse the string 'Hello World'",
        "What's the weather like today?",
        "What's the weather in Louisville?",
        "Search the web for Python programming",
        "Write 'Hello from the AI agent!' to a file called test_output.txt",
        "Read the file test_output.txt",
    ]

    # Multi-turn conversational queries that rely on the agent
    # remembering previous context.
    conversational_queries = [
        "My name is Alex. What time is it?",
        "What is 100 / 5?",
        "What was my name again? And what calculation did I just ask about?",
    ]

    print("\n" + "=" * 60)
    print("Part 1: Running individual tool queries")
    print("=" * 60)

    # Iterate through queries with retry logic and performance tracking
    max_retries = 3
    for query in queries:
        # Validate input
        try:
            validated_query = validate_query(query)
        except InputValidationError as e:
            logger.warning("Input validation failed for query '%s': %s", query, e)
            print(f"\nQuery: {query}")
            print("-" * 50)
            print(f"Validation Error: {e}")
            continue

        print(f"\nQuery: {validated_query}")
        print("-" * 50)

        # Retry logic with exponential backoff
        for attempt in range(1, max_retries + 1):
            start_time = time.time()
            try:
                logger.info(
                    "Sending query (attempt %d/%d): %s",
                    attempt,
                    max_retries,
                    validated_query,
                )
                result = agent.invoke(
                    {"messages": [("user", validated_query)]},
                    config=config,
                )
                duration = time.time() - start_time

                # Extract the final response
                output = result["messages"][-1].content

                # Count tool calls from the messages
                tool_call_count = sum(
                    1 for msg in result["messages"]
                    if hasattr(msg, "tool_calls") and msg.tool_calls
                )

                perf.record_query(validated_query, duration, tool_call_count)
                print(f"Result: {output}")
                break  # Success, exit retry loop

            # Detect HTTP 429 rate limit errors
            except openai.RateLimitError as e:
                duration = time.time() - start_time
                wait_time = 2 ** attempt
                logger.warning(
                    "HTTP 429 rate limit hit on attempt %d/%d, waiting %ds: %s",
                    attempt,
                    max_retries,
                    wait_time,
                    e,
                )
                if attempt < max_retries:
                    print(f"Rate limited (HTTP 429), retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"Rate limit error after {max_retries} attempts: {e}")

            except RateLimitError as e:
                duration = time.time() - start_time
                wait_time = 2 ** attempt
                logger.warning(
                    "Rate limited on attempt %d/%d, waiting %ds: %s",
                    attempt,
                    max_retries,
                    wait_time,
                    e,
                )
                if attempt < max_retries:
                    print(f"Rate limited, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"Rate limit error after {max_retries} attempts: {e}")

            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    "Error on attempt %d/%d for query '%s': %s - %s",
                    attempt,
                    max_retries,
                    validated_query,
                    type(e).__name__,
                    e,
                )
                if attempt < max_retries:
                    wait_time = 2 ** attempt
                    print(f"Error, retrying in {wait_time}s... ({type(e).__name__})")
                    time.sleep(wait_time)
                else:
                    print(f"Error: {type(e).__name__}: {e}")

    # ── Part 2: Conversational Memory Demo ──────────────────────────────────
    print("\n" + "=" * 60)
    print("Part 2: Conversational Memory Demo")
    print("=" * 60)
    print("These queries share a conversation thread, so the agent")
    print("should remember context from previous messages.\n")

    for query in conversational_queries:
        try:
            validated_query = validate_query(query)
        except InputValidationError as e:
            logger.warning("Input validation failed: %s", e)
            print(f"\nQuery: {query}")
            print("-" * 50)
            print(f"Validation Error: {e}")
            continue

        print(f"Query: {validated_query}")
        print("-" * 50)

        for attempt in range(1, max_retries + 1):
            start_time = time.time()
            try:
                logger.info(
                    "Sending conversational query (attempt %d/%d): %s",
                    attempt,
                    max_retries,
                    validated_query,
                )
                # Use the SAME config/thread_id so the agent remembers context
                result = agent.invoke(
                    {"messages": [("user", validated_query)]},
                    config=config,
                )
                duration = time.time() - start_time
                output = result["messages"][-1].content
                tool_call_count = sum(
                    1 for msg in result["messages"]
                    if hasattr(msg, "tool_calls") and msg.tool_calls
                )
                perf.record_query(validated_query, duration, tool_call_count)
                print(f"Result: {output}")
                break

            except openai.RateLimitError as e:
                wait_time = 2 ** attempt
                logger.warning("HTTP 429 rate limit hit: %s", e)
                if attempt < max_retries:
                    print(f"Rate limited (HTTP 429), retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"Rate limit error after {max_retries} attempts: {e}")

            except Exception as e:
                duration = time.time() - start_time
                logger.error("Conversational query error: %s - %s", type(e).__name__, e)
                if attempt < max_retries:
                    wait_time = 2 ** attempt
                    print(f"Error, retrying in {wait_time}s... ({type(e).__name__})")
                    time.sleep(wait_time)
                else:
                    print(f"Error: {type(e).__name__}: {e}")

    # Print performance summary
    print("\n" + "-" * 50)
    print("Performance Summary:")
    print(perf.summary())
    print("\nAgent demo complete!")


if __name__ == "__main__":
    main()
