"""Sanity check: confirm the Anthropic API key in .env works."""

import os

from anthropic import Anthropic
from dotenv import load_dotenv

MODEL = "claude-haiku-5-5"


def main():
    load_dotenv()
    api_key = os.environ["ANTHROPIC_API_KEY"]

    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model=MODEL,
        max_tokens=5,
        messages=[{"role": "user", "content": "Reply with the single word OK"}],
    )

    reply_text = "".join(block.text for block in response.content if block.type == "text")
    print(f"Reply: {reply_text}")
    print(f"Input tokens: {response.usage.input_tokens}")
    print(f"Output tokens: {response.usage.output_tokens}")


if __name__ == "__main__":
    main()
