"""Paper 3: ReAct - Synergizing Reasoning and Acting in Language Models
Yao et al., 2022 - https://arxiv.org/abs/2210.03629

Core idea
---------
Interleave *reasoning* (Thought) with *acting* (Action) and feed the tool's
result back in (Observation). Reasoning decides what to look up; observations
ground the reasoning in real facts instead of hallucinations.

    Thought -> Action -> Observation -> Thought -> Action -> ... -> Final Answer

This loop is the skeleton of nearly every modern agent.
"""
import re

from common import banner, calculate, get_llm

WIKI = {
    "eiffel tower": "The Eiffel Tower in Paris was completed in 1889.",
    "empire state building": "The Empire State Building in New York was completed in 1931.",
}


def search(query):
    return WIKI.get(query.strip().lower(), "No results.")


TOOLS = {"search": search, "calculate": lambda e: str(calculate(e))}

PROMPT = """Answer the question by interleaving Thought, Action and Observation steps.
Available actions:
  search[topic]      - look up a topic in the encyclopedia
  calculate[expr]    - evaluate arithmetic
Write exactly one Thought and one Action per turn, then stop.
When you know the answer, write:  Final Answer: <answer>

Question: {question}
{scratchpad}"""

QUESTION = "How many years after the Eiffel Tower was completed was the Empire State Building completed?"

SCRIPTED = [
    "Thought: I need the completion year of the Eiffel Tower.\nAction: search[Eiffel Tower]",
    "Thought: Eiffel Tower is 1889. Now the Empire State Building.\nAction: search[Empire State Building]",
    "Thought: 1931 and 1889. Subtract them.\nAction: calculate[1931 - 1889]",
    "Thought: The difference is 42 years.\nFinal Answer: 42 years",
]


def react(question, llm, max_steps=6):
    scratchpad = ""
    for step in range(1, max_steps + 1):
        # Stop before "Observation:" so the model can't invent the tool output.
        out = llm(PROMPT.format(question=question, scratchpad=scratchpad), stop=["Observation:"]).strip()
        print(f"--- step {step} ---\n{out}")
        scratchpad += out + "\n"

        final = re.search(r"Final Answer:\s*(.+)", out)
        if final:
            return final.group(1).strip()

        action = re.search(r"Action:\s*(\w+)\[(.*?)\]", out)
        if not action:
            observation = "Invalid action format. Use tool[input]."
        else:
            tool, arg = action.groups()
            observation = TOOLS[tool](arg) if tool in TOOLS else f"Unknown tool {tool}"
        print(f"Observation: {observation}")
        scratchpad += f"Observation: {observation}\n"
    return "gave up"


def main():
    banner("ReAct: " + QUESTION)
    answer = react(QUESTION, get_llm(SCRIPTED))
    print("\nFINAL:", answer)


if __name__ == "__main__":
    main()
