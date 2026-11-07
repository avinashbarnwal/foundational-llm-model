"""Paper 1: Chain-of-Thought Prompting Elicits Reasoning in Large Language Models
Wei et al., 2022 - https://arxiv.org/abs/2201.11903

Core idea
---------
If the few-shot examples in a prompt show the *reasoning steps* before the answer,
the model imitates that and writes its own steps. Spending tokens on intermediate
reasoning sharply improves multi-step math and logic.

    Standard prompt:   Q -> A
    CoT prompt:        Q -> step, step, step -> A

Why it matters for agents: every agent loop below (ReAct, Reflexion, ToT...) is
built on the model "thinking out loud" before it acts.
"""
import re

from common import banner, get_llm

EXAMPLE_Q = (
    "Roger has 5 tennis balls. He buys 2 more cans of tennis balls. "
    "Each can has 3 tennis balls. How many tennis balls does he have now?"
)
QUESTION = (
    "The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, "
    "how many apples do they have?"
)

# The only difference between the two prompts is the worked example's answer.
STANDARD_PROMPT = f"""Q: {EXAMPLE_Q}
A: The answer is 11.

Q: {QUESTION}
A:"""

COT_PROMPT = f"""Q: {EXAMPLE_Q}
A: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls.
5 + 6 = 11. The answer is 11.

Q: {QUESTION}
A:"""

SCRIPTED = [
    # Without reasoning, the paper shows models often pattern-match to a wrong number.
    "The answer is 27.",
    "The cafeteria started with 23 apples. They used 20, so 23 - 20 = 3. "
    "They bought 6 more, so 3 + 6 = 9. The answer is 9.",
]


def final_answer(text):
    match = re.search(r"answer is (-?\d+)", text)
    return match.group(1) if match else "?"


def main():
    llm = get_llm(SCRIPTED)

    banner("Standard few-shot prompt (answer only)")
    out = llm(STANDARD_PROMPT)
    print(out, "\n-> parsed answer:", final_answer(out))

    banner("Chain-of-thought prompt (reasoning, then answer)")
    out = llm(COT_PROMPT)
    print(out, "\n-> parsed answer:", final_answer(out))

    print("\nCorrect answer: 9. Same model, same question; only the example changed.")


if __name__ == "__main__":
    main()
