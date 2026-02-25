import asyncio
from openai import AsyncOpenAI

"""
Industrial Best Practice: Prompt Caching (Prefix Caching)
-------------------------------------------------------
Context: 2025 LLM pricing focuses on "Cached" tokens (Prefix Caching).
Providers like DeepSeek, OpenAI, and Anthropic offer 50-90% discounts
for tokens that stay the same between requests.

Key Strategy:
1. Static Content (System Prompt, Examples, Long Context) MUST be at the beginning.
2. Dynamic Content (User Query, Current Time) MUST be at the end.
"""

# ❌ POOR Practice: Dynamic content at the start breaks the cache for everything after it.
def bad_prompt(query):
    return [
        {"role": "user", "content": f"Today is {time.time()}. Process this: {query}"},
        {"role": "system", "content": "You are a helpful assistant with a 5000-line context..."}
    ]

# ✅ BEST Practice: Static context stays consistent, enabling cache hits.
def good_prompt(query, long_context):
    return [
        {"role": "system", "content": f"You are an expert analyst. Reference Context: {long_context}"},
        # The above part will be cached after the first request
        {"role": "user", "content": f"Query: {query}"}
    ]

async def call_with_caching(client: AsyncOpenAI, messages: list):
    """
    Note: For DeepSeek/Anthropic, caching is often automatic if the prefix matches.
    Some providers require a 'cache_control' parameter.
    """
    res = await client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        # extra_body={"cache_control": {"type": "ephemeral"}} # Example for Anthropic
    )
    # Check for cache hit in metadata (if supported)
    # print(res.usage.prompt_tokens_details.cached_tokens)
    return res

if __name__ == "__main__":
    print("Prompt Caching Pattern guide loaded.")
    print("Rule: Keep the 'Fixed Prefix' as long and stable as possible.")
