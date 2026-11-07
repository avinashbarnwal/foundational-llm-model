"""Paper 8: Tree of Thoughts - Deliberate Problem Solving with Large Language Models
Yao et al., 2023 - https://arxiv.org/abs/2305.10601

Core idea
---------
Chain-of-thought commits to one line of reasoning. ToT treats each partial
solution ("thought") as a node in a tree and runs a search over it:

    propose(state)  -> several candidate next thoughts       (LLM in the paper)
    evaluate(state) -> how promising is it? sure/likely/impossible (LLM in the paper)
    search          -> BFS keeping the best b states per level (or DFS + backtrack)

Demo task, from the paper: Game of 24. Combine 4 numbers with + - * / to make 24.
A thought is one step, e.g. "13 - 9 = 4 (left: 4 4 10)".

Offline, propose() enumerates moves and evaluate() is a heuristic so it runs
instantly. With --live, evaluate() asks Claude the paper's value prompt.
"""
from itertools import combinations

from common import banner, get_llm, is_live

TARGET = 24
NUMBERS = [4, 9, 10, 13]


def fmt(x):
    return str(int(x)) if float(x).is_integer() else f"{x:.3g}"


def propose(state):
    """All ways to replace two numbers with their combination."""
    nums, history = state
    children = []
    for i, j in combinations(range(len(nums)), 2):
        a, b = nums[i], nums[j]
        rest = [n for k, n in enumerate(nums) if k not in (i, j)]
        moves = [(a + b, "+", a, b), (a * b, "*", a, b), (a - b, "-", a, b), (b - a, "-", b, a)]
        if b:
            moves.append((a / b, "/", a, b))
        if a:
            moves.append((b / a, "/", b, a))
        for value, op, x, y in moves:
            new = sorted(rest + [value])
            step = f"{fmt(x)} {op} {fmt(y)} = {fmt(value)} (left: {' '.join(fmt(n) for n in new)})"
            children.append((new, history + [step]))
    return children


def solvable(nums):
    if len(nums) == 1:
        return abs(nums[0] - TARGET) < 1e-6
    return any(solvable(child) for child, _ in propose((nums, [])))


def heuristic_value(nums):
    """Offline stand-in for the LLM's sure / impossible judgement.

    The paper's LLM "looks ahead" a few steps in its head; here we do that
    lookahead exactly. Same role in the search, just a perfect evaluator.
    """
    return 1.0 if solvable(nums) else 0.001


VALUE_PROMPT = """Evaluate if the given numbers can reach 24 using + - * / (each number used once).
Reply with one word: sure, likely, or impossible.
Numbers: {nums}"""
VALUE_MAP = {"sure": 20.0, "likely": 1.0, "impossible": 0.001}  # weights from the paper


def make_llm_value(llm):
    def value(nums):
        if len(nums) == 1:
            return heuristic_value(nums)
        word = llm(VALUE_PROMPT.format(nums=" ".join(fmt(n) for n in nums))).strip().lower()
        return VALUE_MAP.get(word.split()[0] if word else "", 0.001)

    return value


def tree_of_thoughts(numbers, value_fn, beam=5):
    frontier = [(sorted(numbers), [])]
    for depth in range(1, len(numbers)):
        candidates = [c for s in frontier for c in propose(s)]
        scored = sorted(candidates, key=lambda c: value_fn(c[0]), reverse=True)
        frontier = scored[:beam]
        print(f"depth {depth}: {len(candidates):>3} thoughts proposed, keeping {len(frontier)}")
        for nums, hist in frontier:
            print("    " + hist[-1])
    for nums, hist in frontier:
        if abs(nums[0] - TARGET) < 1e-6:
            return hist
    return None


def main():
    banner(f"Tree of Thoughts on Game of 24 with {NUMBERS}")
    value_fn = make_llm_value(get_llm([])) if is_live() else heuristic_value
    solution = tree_of_thoughts(NUMBERS, value_fn)
    print("\nSolution path:" if solution else "\nNo solution found.")
    for step in solution or []:
        print("  " + step)


if __name__ == "__main__":
    main()
