
import anthropic
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
open_ai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
deepseek_client = OpenAI( api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com" )
claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def call_openai_chat(model, prompt, return_usage=False):
    # model="o3-mini-2025-01-31"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    completion = open_ai_client.chat.completions.create(
        model=model,
        messages=messages
    )

    content = completion.choices[0].message.content
    print(content)

    if return_usage:
        return content, completion.usage.model_dump(), completion
    return content

def call_deepseek_chat(model, prompt, return_usage=False):
    # model="deepseek-reasoner",
    messages = [
        {"role": "user", "content": prompt}
    ]

    completion = deepseek_client.chat.completions.create(
        model=model,
        max_tokens=1400,
        messages=messages
    )

    content = completion.choices[0].message.content
    reasoning = completion.choices[0].message.reasoning_content

    print("=== REASONING CONTENT ===")
    print(reasoning)
    print("=== FINAL ANSWER ===")
    print(content)

    if return_usage:
        return content, {}, reasoning
    return content



def call_anthropic_claude(model, prompt, return_usage=False):
    # model="claude-3-7-sonnet-20250219",

    response = claude_client.messages.create(
        model=model,
        max_tokens=1400,
        thinking={
            "type": "enabled",
            "budget_tokens": 1024
        },
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    thinking = next((b.thinking for b in response.content if b.type == "thinking"), None)
    text = next((b.text for b in response.content if b.type == "text"), None)

    print("=== CLAUDE REASONING (thinking) ===")
    print(thinking if thinking else "[No reasoning output]")
    print("=== CLAUDE FINAL RESPONSE ===")
    print(text)

    if return_usage:
        return text, {}, thinking if thinking else text
    return text






