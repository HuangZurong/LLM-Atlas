import yaml
import jinja2
from typing import Dict, Any, List

"""
Industrial Prompt Infrastructure (Advanced)
------------------------------------------
Features:
1. Jinja2 Templating: Supports loops, conditionals, and complex logic inside prompts.
2. Unit Testing: Automated validation of prompt rendering.
3. Separation of Concerns: Environment-agnostic prompt definitions.
"""

PROMPT_LIBRARY = """
reasoning_agent:
  version: "1.2.0"
  system: |
    You are a reasoning assistant.
    Current Date: {{ current_date }}
    {% if strict_mode %}
    Follow these rules strictly:
    1. Only use the provided tools.
    2. Do not speculate.
    {% endif %}
  user_template: |
    User query: {{ query }}
    Context:
    {% for item in context_list %}
    - {{ item }}
    {% endfor %}
"""

class PromptTemplate:
    def __init__(self, raw_data: Dict):
        self.version = raw_data.get('version', '0.0.1')
        self.env = jinja2.Environment(undefined=jinja2.StrictUndefined)
        self.system_tmpl = self.env.from_string(raw_data['system'])
        self.user_tmpl = self.env.from_string(raw_data['user_template'])
        self.config = raw_data.get('config', {})

    def render(self, **kwargs) -> List[Dict]:
        return [
            {"role": "system", "content": self.system_tmpl.render(**kwargs)},
            {"role": "user", "content": self.user_tmpl.render(**kwargs)}
        ]

class PromptManager:
    def __init__(self):
        data = yaml.safe_load(PROMPT_LIBRARY)
        self.templates = {k: PromptTemplate(v) for k, v in data.items()}

    def get_prompt(self, name: str) -> PromptTemplate:
        return self.templates[name]

# ───────────────────────────────────────────────────────────────────────────
# Industrial Pattern: Prompt Unit Testing
# ───────────────────────────────────────────────────────────────────────────
def test_prompt_rendering():
    pm = PromptManager()
    tmpl = pm.get_prompt("reasoning_agent")

    # Test case 1: Strict mode with context
    msgs = tmpl.render(
        current_date="2026-02-24",
        strict_mode=True,
        query="What is the weather?",
        context_list=["Sensor A: Sunny", "Sensor B: 25C"]
    )

    assert "Follow these rules strictly" in msgs[0]['content']
    assert "- Sensor A: Sunny" in msgs[1]['content']
    print("✓ Prompt Unit Test Passed: Complex Rendering")

if __name__ == "__main__":
    test_prompt_rendering()

    # Usage
    pm = PromptManager()
    reasoner = pm.get_prompt("reasoning_agent")
    payload = reasoner.render(
        current_date="2026-02-24",
        strict_mode=False,
        query="Simple task",
        context_list=[]
    )
    print("\nRendered System Prompt (Strict Mode Off):")
    print(payload[0]['content'])
