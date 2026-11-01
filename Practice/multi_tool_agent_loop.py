from dotenv import load_dotenv
from anthropic import Anthropic
from anthropic.types import Message
import json

load_dotenv()

client = Anthropic()

model = "claude-haiku-4-5"

# Tool Functions

def calculator(expression):
    # Simple calculator function that evaluates a mathematical expression
    try:
        result = eval(expression, {"__builtins__": None}, {})
        return str(result)
    except Exception as e:
        return f"Error: Invalid expression - {e}"

def web_search(query):
    # Mock search: returns results in the format the tool description promises
    results = [
        {
            "title": f"Bitcoin price today (search: {query})",
            "snippet": "Bitcoin is trading at $65,000 USD, up 2% over the past 24 hours."
        },
        {
            "title": "Crypto market overview",
            "snippet": "Major cryptocurrencies saw modest gains this week."
        }
    ]
    return json.dumps(results)

# Tool Schemas

calculator_tool_schema = {
    "name": "calculator",
    "description": (
        "Evaluates an arithmetic expression and returns the exact numeric result. "
        "Use this whenever a calculation is needed, rather than working it out in your head, "
        "for example multiplying a price by a quantity or computing a percentage. "
        "Supports +, -, *, /, ** (power), % (modulo), // (floor division), parentheses and decimals. "
        "Expressions must use Python syntax: write powers as 2 ** 3, not 2 ^ 3. "
        "It does not support variables, units, currency symbols or thousands separators, "
        "so pass plain numbers only (65000, not $65,000). "
        "If the expression cannot be evaluated, the tool returns a string starting with 'Error:'."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": (
                    "A single arithmetic expression in Python syntax, using plain numbers only. "
                    "Examples: '65000 * 3.5', '(120 - 45) / 3', '2 ** 10'."
                ),
                "minLength": 1
            }
        },
        "required": ["expression"]
    }
}

web_search_tool_schema = {
    "name": "web_search", 
    "description": (
        "Searches the web for information and returns relevant results each with a title and a short snippet. "
        "Use this tool to find up-to-date information, news, or specific details that are not available in the model's training data."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "A search query string to find relevant information on the web. "
                    "Examples: 'latest news on climate change', 'Python programming tutorials', 'current stock prices for AAPL'."
                ),
                "minLength": 1
            }
        },
        "required": ["query"]
    }
}

tool_list=[calculator_tool_schema, web_search_tool_schema]

#Run the agent
import logging
MAX_ITERATIONS = 20
def run_agent(user_prompt):
    messages = [{"role": "user", "content": user_prompt}]
    iterations = 0
    while True:
        if iterations >= MAX_ITERATIONS:
            logging.warning("Safety cap of %d iterations reached; stopping the agent.", MAX_ITERATIONS)
            return "Agent stopped: safety iteration cap reached before the task finished."
        iterations += 1
            
        response = client.messages.create(
            model=model,
            max_tokens=1000,
            messages=messages,
            tools=tool_list,
        )

        # Print any text Claude wrote, wherever it appears in the response
        for block in response.content:
            if block.type == "text":
                print(f"Assistant: {block.text}")

        if response.stop_reason == "end_turn":
            # Step 4: extract the final text and return it 
            print(f"(Finished in {iterations} iterations)") 
            return "\n".join([block.text for block in response.content if block.type == "text"])
            

        if response.stop_reason == "tool_use":
            # 1. Append Claude's full response (text and tool_use blocks) to the history
            messages.append({"role": "assistant", "content": response.content})

            # 2. Run every tool Claude asked for, collecting one result per request
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"Using tool: {block.name} with input: {block.input}")

                    if block.name == "calculator":
                        tool_output = calculator(block.input["expression"])
                    elif block.name == "web_search":
                        tool_output = web_search(block.input["query"])
                    else:
                        tool_output = f"Error: Unknown tool '{block.name}'"

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,   # links this result to its request
                        "content": tool_output,
                    })

            # 3. Send all results back in a single user message
            messages.append({"role": "user", "content": tool_results})
            # 4. The while loop now repeats with the updated history
            # messages=[{"role": "user", "content": user_prompt}]
    


answer = run_agent("Search for the current price of Bitcoin and calculate what 3.5 coins would cost")
print("\nFINAL ANSWER:", answer)

