"""
Structured Output Patterns — Production implementations.

Demonstrates:
1. Pydantic-based structured extraction with fallback.
2. Multi-tool dispatcher with automatic routing.
"""

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, ValidationError


# ── 1. Pydantic Schema Definition ──────────────────────────────────

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class ReviewAnalysis(BaseModel):
    """Schema for product review analysis."""
    sentiment: Sentiment = Field(description="Overall sentiment of the review.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score.")
    key_topics: list[str] = Field(description="Main topics, max 5.", max_length=5)
    summary: str = Field(description="One-sentence summary.")


# ── 2. Structured Output Caller (with fallback) ───────────────────

def extract_structured(text: str, schema: type[BaseModel], model: str = "gpt-4o") -> BaseModel:
    """
    Extract structured data using OpenAI's Structured Outputs.
    Falls back to JSON mode + manual validation if strict mode fails.
    """
    from openai import OpenAI
    client = OpenAI()

    # Attempt 1: Strict structured output (constrained decoding)
    try:
        resp = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": "Extract the requested information from the text."},
                {"role": "user", "content": text},
            ],
            response_format=schema,
        )
        if resp.choices[0].message.refusal:
            raise ValueError(f"Model refused: {resp.choices[0].message.refusal}")
        return resp.choices[0].message.parsed
    except Exception:
        pass

    # Attempt 2: JSON mode + Pydantic validation
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": f"Reply in JSON matching this schema:\n{schema.model_json_schema()}"},
            {"role": "user", "content": text},
        ],
        response_format={"type": "json_object"},
    )
    raw = json.loads(resp.choices[0].message.content)
    return schema.model_validate(raw)


# ── 3. Multi-Tool Dispatcher ───────────────────────────────────────

@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: callable


class ToolDispatcher:
    """
    Registers tools and dispatches LLM tool_calls to the correct handler.
    """

    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.tools: dict[str, ToolDefinition] = {}

    def register(self, name: str, description: str, parameters: dict, handler: callable):
        self.tools[name] = ToolDefinition(name, description, parameters, handler)

    def _tool_schemas(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self.tools.values()
        ]

    def run(self, user_message: str) -> str:
        from openai import OpenAI
        client = OpenAI()
        messages = [{"role": "user", "content": user_message}]

        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self._tool_schemas(),
            tool_choice="auto",
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            return msg.content

        # Execute each tool call and collect results
        messages.append(msg)
        for tc in msg.tool_calls:
            handler = self.tools[tc.function.name].handler
            args = json.loads(tc.function.arguments)
            result = handler(**args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, ensure_ascii=False),
            })

        final = client.chat.completions.create(model=self.model, messages=messages)
        return final.choices[0].message.content


# ── 4. Usage Example ───────────────────────────────────────────────

if __name__ == "__main__":
    # Structured extraction
    review = "The battery life is amazing but the camera quality is disappointing."
    try:
        analysis = extract_structured(review, ReviewAnalysis)
        print(f"Sentiment: {analysis.sentiment}, Topics: {analysis.key_topics}")
    except (ValidationError, Exception) as e:
        print(f"Extraction failed: {e}")

    # Tool dispatcher
    dispatcher = ToolDispatcher()
    dispatcher.register(
        name="get_weather",
        description="Get current weather for a city.",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
        handler=lambda city: {"city": city, "temp": 22, "condition": "sunny"},
    )
    print(dispatcher.run("What's the weather in Beijing?"))