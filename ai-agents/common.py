"""Shared helpers for the paper demos.

Every demo takes an `llm` callable: llm(prompt, system="", stop=None) -> str.

- By default it is a ScriptedLLM that replays canned replies, so each demo runs
  offline and you can read the mechanism without an API key.
- Pass --live to any demo to use Claude instead (needs `pip install anthropic`
  and ANTHROPIC_API_KEY, or an `ant auth login` profile).
"""
import ast
import operator
import sys

MODEL = "claude-opus-5-5"


class ScriptedLLM:
    """Offline stand-in that returns canned replies in order."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def __call__(self, prompt, system="", stop=None):
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return reply


class ClaudeLLM:
    """Real backend: one Messages API call per invocation."""

    def __init__(self, model=MODEL):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model

    def __call__(self, prompt, system="", stop=None):
        kwargs = dict(
            model=self.model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
            betas=["server-side-fallback-2026-07-01"],
            # fallbacks: retry on another model if a safety classifier declines.
            # effort low: these are tiny demo tasks.
            extra_body={"fallbacks": "default", "output_config": {"effort": "low"}},
        )
        if system:
            kwargs["system"] = system
        if stop:
            kwargs["stop_sequences"] = stop
        resp = self.client.beta.messages.create(**kwargs)
        if resp.stop_reason == "refusal":
            raise RuntimeError("Claude declined this request.")
        return "".join(b.text for b in resp.content if b.type == "text")


def get_llm(scripted_replies):
    """Claude with --live, otherwise the scripted replies."""
    if "--live" in sys.argv:
        return ClaudeLLM()
    return ScriptedLLM(scripted_replies)


def is_live():
    return "--live" in sys.argv


_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def calculate(expr):
    """Safe arithmetic evaluator (no eval): numbers and + - * / ** only."""

    def walk(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.operand))
        raise ValueError("unsupported expression: " + expr)

    result = walk(ast.parse(expr, mode="eval").body)
    return round(result, 4) if isinstance(result, float) else result


def extract_code(text):
    """Return the first ```python fenced block, or the text itself."""
    if "```" not in text:
        return text.strip()
    block = text.split("```", 2)[1]
    if block.startswith("python"):
        block = block[len("python"):]
    return block.strip()


def banner(title):
    print("\n" + "=" * 70 + "\n" + title + "\n" + "=" * 70)
