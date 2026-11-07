"""Paper 6: Self-Refine - Iterative Refinement with Self-Feedback
Madaan et al., 2023 - https://arxiv.org/abs/2303.17651

Core idea
---------
One model plays three roles in a loop, with no training and no external tools:

    draft = GENERATE(task)
    loop:
        feedback = FEEDBACK(task, draft)      # specific, actionable critique
        if feedback says it's good: stop
        draft = REFINE(task, draft, feedback)

Difference from Reflexion: no external evaluator; the model critiques itself,
and refinement happens within one task instead of across trials.
"""
from common import banner, get_llm

TASK = "Write a one-sentence product description for a reusable water bottle."

GENERATE = "{task}"

FEEDBACK = """Task: {task}
Draft: {draft}

Critique the draft on: specificity, vivid detail, and a clear benefit to the buyer.
Give concrete suggestions. If it is already excellent on all three, reply only: STOP"""

REFINE = """Task: {task}
Draft: {draft}
Feedback: {feedback}

Rewrite the draft applying the feedback. Reply with only the new sentence."""

SCRIPTED = [
    "This is a good water bottle that you can use again.",
    "Vague: 'good' says nothing. Add a concrete feature (material, insulation) and a "
    "buyer benefit (cold drinks, less plastic waste).",
    "This stainless-steel bottle keeps water cold for 24 hours and replaces hundreds of plastic bottles.",
    "Better. Make it more vivid, e.g. mention a real moment of use, and keep the eco benefit.",
    "From a sunrise hike to a late-night study session, this stainless-steel bottle keeps "
    "water ice-cold for 24 hours while sparing the planet hundreds of plastic bottles.",
    "STOP",
]


def self_refine(llm, max_rounds=4):
    draft = llm(GENERATE.format(task=TASK)).strip()
    print("draft 0:", draft)
    for i in range(1, max_rounds + 1):
        feedback = llm(FEEDBACK.format(task=TASK, draft=draft)).strip()
        if feedback.upper().startswith("STOP"):
            print("\nfeedback: STOP (model is satisfied)")
            break
        print(f"\nfeedback {i}: {feedback}")
        draft = llm(REFINE.format(task=TASK, draft=draft, feedback=feedback)).strip()
        print(f"draft {i}: {draft}")
    return draft


def main():
    banner("Self-Refine: " + TASK)
    final = self_refine(get_llm(SCRIPTED))
    print("\nFINAL:", final)


if __name__ == "__main__":
    main()
