"""
hello.py — the smallest possible Claude script.

Goal: prove your API key works and the anthropic SDK is installed correctly,
BEFORE we add agent loops, tools, or any other complexity.

Run it from your terminal with:
    python3 hello.py
"""

import os
from dotenv import load_dotenv
from anthropic import Anthropic

# Load the ANTHROPIC_API_KEY from your .env file into os.environ
load_dotenv()

# Create the client. It picks up ANTHROPIC_API_KEY from the environment automatically.
client = Anthropic()

# Send a single message and get a response back.
response = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=300,
    messages=[
        {"role": "user", "content": "In two sentences, what is an AI agent?"}
    ],
)

# The response contains a list of content blocks. For a plain text reply, just grab the first one.
print(response.content[0].text)
