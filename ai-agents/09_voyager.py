"""Paper 9: Voyager - An Open-Ended Embodied Agent with Large Language Models
Wang et al. (NVIDIA/Caltech), 2023 - https://arxiv.org/abs/2305.16291

Core idea
---------
A lifelong-learning agent (in Minecraft) built from three parts:

1. Automatic curriculum - the LLM proposes the next task based on what the
   agent has and has already done ("explore as much as possible").
2. Iterative prompting - the LLM writes *code* for the task; the environment
   runs it and returns errors/feedback; the LLM fixes it until it succeeds.
3. Skill library - every verified program is saved as a reusable skill and
   retrieved for later tasks, so abilities compound instead of being re-derived.

Here the "world" is a tiny crafting game instead of Minecraft.
"""
from common import banner, extract_code, get_llm

RECIPES = {  # item: ingredients needed
    "planks": {"log": 1},
    "stick": {"planks": 1},
    "wooden_pickaxe": {"planks": 3, "stick": 2},
    "stone_pickaxe": {"stone": 3, "stick": 2},
}
NEEDS_TOOL = {"stone": "wooden_pickaxe", "iron_ore": "stone_pickaxe"}


class World:
    def __init__(self):
        self.inventory = {}

    def mine(self, item, n=1):
        tool = NEEDS_TOOL.get(item)
        if tool and not self.inventory.get(tool):
            raise RuntimeError(f"cannot mine {item}: need {tool}")
        self.inventory[item] = self.inventory.get(item, 0) + n

    def craft(self, item, n=1):
        for _ in range(n):
            for ingredient, qty in RECIPES[item].items():
                if self.inventory.get(ingredient, 0) < qty:
                    raise RuntimeError(f"cannot craft {item}: need {qty} {ingredient}")
            for ingredient, qty in RECIPES[item].items():
                self.inventory[ingredient] -= qty
            self.inventory[item] = self.inventory.get(item, 0) + 1


CURRICULUM_PROMPT = """You are an explorer in a crafting game. Propose the single next task
that builds on what you can already do. Reply with only the target item name.
Inventory: {inventory}
Completed tasks: {done}
Known recipes: {recipes}  Mining requires: {tools}"""

CODE_PROMPT = """Write a Python function `{fn}(world)` that ends with at least one {item} in world.inventory.
Available: world.mine(item, n), world.craft(item, n), and these existing skills (call them freely):
{skills}
Recipes: {recipes}  Mining requires: {tools}
{feedback}
Reply with only a ```python code block."""

SCRIPTED = [
    # task 1: curriculum, then code
    "wooden_pickaxe",
    "```python\ndef get_wooden_pickaxe(world):\n    world.mine('log', 3)\n"
    "    world.craft('planks', 3)\n    world.craft('wooden_pickaxe')\n```",
    # first attempt fails (forgot sticks); environment feedback drives a fix
    "```python\ndef get_wooden_pickaxe(world):\n    world.mine('log', 5)\n    world.craft('planks', 5)\n"
    "    world.craft('stick', 2)\n    world.craft('wooden_pickaxe')\n```",
    # task 2 reuses the skill from task 1
    "stone_pickaxe",
    "```python\ndef get_stone_pickaxe(world):\n    if not world.inventory.get('wooden_pickaxe'):\n"
    "        get_wooden_pickaxe(world)\n    world.mine('log', 2)\n    world.craft('planks', 2)\n"
    "    world.craft('stick', 2)\n    world.mine('stone', 3)\n"
    "    world.craft('stone_pickaxe')\n```",
    # task 3 reuses the skill from task 2
    "iron_ore",
    "```python\ndef get_iron_ore(world):\n    if not world.inventory.get('stone_pickaxe'):\n"
    "        get_stone_pickaxe(world)\n    world.mine('iron_ore', 1)\n```",
]


def voyager(llm, n_tasks=3, max_retries=3):
    world, skills, done = World(), {}, []
    for _ in range(n_tasks):
        item = llm(CURRICULUM_PROMPT.format(
            inventory=world.inventory, done=done, recipes=RECIPES, tools=NEEDS_TOOL)).strip()
        fn = f"get_{item}"
        banner(f"Curriculum proposes: {item}")

        feedback = ""
        for attempt in range(1, max_retries + 1):
            code = extract_code(llm(CODE_PROMPT.format(
                fn=fn, item=item, recipes=RECIPES, tools=NEEDS_TOOL, feedback=feedback,
                skills="\n\n".join(skills.values()) or "(none yet)")))
            print(f"attempt {attempt}:\n{code}")

            trial = World()
            trial.inventory = dict(world.inventory)
            namespace = {}
            try:
                for skill_code in skills.values():
                    exec(skill_code, namespace)
                exec(code, namespace)
                namespace[fn](trial)
                if not trial.inventory.get(item):
                    raise RuntimeError(f"finished but no {item} in inventory")
            except Exception as e:  # noqa: BLE001 - errors become feedback to the LLM
                feedback = f"Your previous code failed with: {e}. Fix it."
                print(f"  environment feedback -> {e}\n")
                continue

            world = trial
            skills[fn] = code  # verified -> add to the skill library
            done.append(item)
            print(f"  success, skill '{fn}' saved. inventory={world.inventory}")
            break

    banner("Skill library")
    print("\n".join(skills))


def main():
    voyager(get_llm(SCRIPTED))


if __name__ == "__main__":
    main()
