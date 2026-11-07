"""Paper 10: AutoGen - Enabling Next-Gen LLM Applications via Multi-Agent Conversation
Wu et al. (Microsoft), 2023 - https://arxiv.org/abs/2308.08155

Core idea
---------
Build applications as a *conversation between agents*, each a "conversable
agent" with a role. The canonical pair:

    AssistantAgent  (LLM)   - plans and writes code
    UserProxyAgent  (tool)  - executes the code and replies with the output

They message each other automatically until someone says TERMINATE. Adding more
agents (critic, planner, human) is just adding more participants to the chat.
"""
import subprocess
import sys

from common import banner, extract_code, get_llm

ASSISTANT_SYSTEM = """You are a helpful AI assistant. Solve tasks by writing Python code in a
```python block; the user will run it and send you the output. When the task is
fully solved, give the final answer and end your message with TERMINATE."""

TASK = "What are the first 10 Fibonacci numbers, and what is their sum?"

SCRIPTED = [
    "I'll compute them with a short script.\n```python\nfib = [0, 1]\n"
    "while len(fib) < 10:\n    fib.append(fib[-1] + fib[-2])\nprint(fib)\nprint(sum(fib))\n```",
    "The first 10 Fibonacci numbers are 0, 1, 1, 2, 3, 5, 8, 13, 21, 34 and their sum is 88.\nTERMINATE",
]


class AssistantAgent:
    def __init__(self, name, llm, system):
        self.name, self.llm, self.system = name, llm, system

    def reply(self, history):
        transcript = "\n\n".join(f"[{who}]\n{msg}" for who, msg in history)
        return self.llm(transcript + f"\n\n[{self.name}]\n", system=self.system)


class UserProxyAgent:
    """Runs any code block it receives, in a subprocess, and reports the result."""

    def __init__(self, name):
        self.name = name

    def reply(self, history):
        last = history[-1][1]
        if "```" not in last:
            return "No code to run. Continue, or reply TERMINATE if done."
        code = extract_code(last)
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
        status = "succeeded" if proc.returncode == 0 else "failed"
        return f"exitcode: {proc.returncode} (execution {status})\nOutput:\n{proc.stdout}{proc.stderr}"


def initiate_chat(sender, receiver, message, max_turns=8):
    history = [(sender.name, message)]
    print(f"[{sender.name}] {message}\n")
    speakers = [receiver, sender]
    for turn in range(max_turns):
        agent = speakers[turn % 2]
        msg = agent.reply(history)
        history.append((agent.name, msg))
        print(f"[{agent.name}]\n{msg}\n")
        if "TERMINATE" in msg:
            break
    return history


def main():
    assistant = AssistantAgent("assistant", get_llm(SCRIPTED), ASSISTANT_SYSTEM)
    user_proxy = UserProxyAgent("user_proxy")
    banner("AutoGen two-agent chat")
    initiate_chat(user_proxy, assistant, TASK)


if __name__ == "__main__":
    main()
