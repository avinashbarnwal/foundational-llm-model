"""Paper 4: Toolformer - Language Models Can Teach Themselves to Use Tools
Schick et al. (Meta), 2023 - https://arxiv.org/abs/2302.04761

Core idea
---------
Teach the model to write API calls *inline* in its own text, e.g.

    "Out of 1400 participants, 400 [Calculator(400/1400) -> 0.29] or 29% passed."

Two parts, both shown below:

1. Inference: generate until the model emits "->", pause, run the API, splice
   the result in, and let the model continue with the real value.
2. Self-supervised training data: sample candidate API calls in plain text and
   KEEP a call only if inserting its result makes the following tokens easier to
   predict (loss drops by at least tau). No human labels needed.
"""
import re

from common import banner, calculate, get_llm

TOOLS = {
    "Calculator": lambda arg: str(calculate(arg)),
    "Calendar": lambda arg: "Today is Sunday, October 25, 2026.",
}

# ---------------------------------------------------------------------------
# Part 1: inference with inline calls
# ---------------------------------------------------------------------------
PROMPT = """Continue the text. When a fact or number needs a tool, write
[Tool(input) -> and stop; the result will be filled in for you.
Tools: Calculator(expr), Calendar()

Text: Out of 1400 participants, 400"""

SCRIPTED = [
    " [Calculator(400/1400) ->",
    "] or about 29% passed the test.",
]


def generate_with_tools(llm, prompt, max_calls=5):
    text = ""
    for _ in range(max_calls + 1):
        chunk = llm(prompt + text, stop=["->"])
        text += chunk
        call = re.search(r"\[(\w+)\((.*?)\)\s*(->)?\s*$", text)
        if not call or call.group(1) not in TOOLS:
            return text  # no pending call: generation finished
        name, arg = call.group(1), call.group(2)
        result = TOOLS[name](arg)
        print(f"  paused at {name}({arg}) -> executed -> {result}")
        text = text.rstrip().rstrip("->").rstrip() + f" -> {result}"
    return text


# ---------------------------------------------------------------------------
# Part 2: the self-supervised filter
# ---------------------------------------------------------------------------
def toy_loss(prefix, continuation):
    """Stand-in for the LM's loss on `continuation` given `prefix`.

    The real paper uses the model's cross-entropy. Here: the continuation is
    "easy" if the numbers it contains already appear in the prefix.
    """
    numbers = re.findall(r"\d+(?:\.\d+)?", continuation)
    if not numbers:
        return 1.0
    seen = sum(n in prefix for n in numbers)
    return 1.0 - 0.8 * seen / len(numbers)


def keep_call(prefix, call, result, continuation, tau=0.3):
    loss_without = min(toy_loss(prefix, continuation), toy_loss(prefix + f" [{call}]", continuation))
    loss_with = toy_loss(prefix + f" [{call} -> {result}]", continuation)
    keep = loss_without - loss_with >= tau
    print(f"  {call:<28} L-={loss_without:.2f}  L+={loss_with:.2f}  keep={keep}")
    return keep


def main():
    llm = get_llm(SCRIPTED)

    banner("Part 1 - inference: pause, call tool, continue")
    out = generate_with_tools(llm, PROMPT)
    print("\nFinal text: Out of 1400 participants, 400" + out)

    banner("Part 2 - which sampled API calls become training data?")
    prefix = "The bill was 85 dollars and we split it among 4 friends, so each paid"
    continuation = " 21.25 dollars."
    for call in ["Calculator(85/4)", "Calculator(85*4)", "Calendar()"]:
        name, arg = call.split("(", 1)
        keep_call(prefix, call, TOOLS[name](arg.rstrip(")")), continuation)
    print("\nOnly the call whose result helps predict ' 21.25' is kept for fine-tuning.")


if __name__ == "__main__":
    main()
