"""Paper 5: Reflexion - Language Agents with Verbal Reinforcement Learning
Shinn et al., 2023 - https://arxiv.org/abs/2303.11366

Core idea
---------
Instead of updating weights, the agent learns from failure in *words*:

    attempt -> evaluator (e.g. unit tests) -> fail?
            -> self-reflection: "I failed because ... next time I should ..."
            -> store reflection in episodic memory
            -> retry with the reflections in the prompt

The reflection is a cheap, interpretable "gradient" carried across trials.
"""
from common import banner, extract_code, get_llm

TASK = (
    "Write a Python function is_palindrome(s) that returns True if s reads the same "
    "forwards and backwards, ignoring case, spaces and punctuation."
)

TESTS = [
    ("racecar", True),
    ("A man, a plan, a canal: Panama", True),
    ("hello", False),
    ("No 'x' in Nixon", True),
]

ACT_PROMPT = """{task}

Lessons from your previous failed attempts:
{memory}

Reply with only a ```python code block."""

REFLECT_PROMPT = """You wrote this code for the task below and it failed.
Task: {task}
Code:
{code}
Failures:
{failures}

In 2-3 sentences, say why it failed and what to do differently next time."""

SCRIPTED = [
    "```python\ndef is_palindrome(s):\n    return s == s[::-1]\n```",
    "My solution compared the raw string, so capitals, spaces and punctuation broke it. "
    "Next time I should lowercase the string and keep only alphanumeric characters first.",
    "```python\ndef is_palindrome(s):\n    t = [c.lower() for c in s if c.isalnum()]\n    return t == t[::-1]\n```",
]


def evaluate(code):
    """The evaluator: run unit tests, return the list of failures."""
    namespace = {}
    try:
        exec(code, namespace)
        fn = namespace["is_palindrome"]
    except Exception as e:  # noqa: BLE001 - any error is feedback for the agent
        return [f"code did not run: {e!r}"]
    return [
        f"is_palindrome({arg!r}) returned {fn(arg)!r}, expected {want!r}"
        for arg, want in TESTS
        if fn(arg) != want
    ]


def reflexion(llm, max_trials=3):
    memory = []
    for trial in range(1, max_trials + 1):
        banner(f"Trial {trial}")
        lessons = "\n".join(f"- {m}" for m in memory) or "(none yet)"
        code = extract_code(llm(ACT_PROMPT.format(task=TASK, memory=lessons)))
        print(code)

        failures = evaluate(code)
        if not failures:
            print("\nAll tests passed.")
            return code
        print("\nFailed tests:\n  " + "\n  ".join(failures))

        reflection = llm(REFLECT_PROMPT.format(task=TASK, code=code, failures="\n".join(failures)))
        print("\nSelf-reflection:", reflection)
        memory.append(reflection.strip())
    return None


def main():
    reflexion(get_llm(SCRIPTED))


if __name__ == "__main__":
    main()
