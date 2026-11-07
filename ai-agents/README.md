# 10 Foundational AI Agent Papers, in Code

Each file is a small, runnable demo that implements the core mechanism of one paper.
Read the docstring at the top of each file first, then run it.

| # | File | Paper | The one idea |
|---|------|-------|--------------|
| 1 | `01_chain_of_thought.py` | [Chain-of-Thought](https://arxiv.org/abs/2201.11903) (Wei et al., 2022) | Show reasoning steps in examples so the model writes its own steps |
| 2 | `02_mrkl.py` | [MRKL Systems](https://arxiv.org/abs/2205.00445) (Karpas et al., 2022) | LLM as a router that sends each query to the right expert module |
| 3 | `03_react.py` | [ReAct](https://arxiv.org/abs/2210.03629) (Yao et al., 2022) | Thought → Action → Observation loop |
| 4 | `04_toolformer.py` | [Toolformer](https://arxiv.org/abs/2302.04761) (Schick et al., 2023) | Inline API calls, kept as training data only if they lower the loss |
| 5 | `05_reflexion.py` | [Reflexion](https://arxiv.org/abs/2303.11366) (Shinn et al., 2023) | Learn from failure through written self-reflections stored in memory |
| 6 | `06_self_refine.py` | [Self-Refine](https://arxiv.org/abs/2303.17651) (Madaan et al., 2023) | Generate → self-feedback → refine, repeated |
| 7 | `07_generative_agents.py` | [Generative Agents](https://arxiv.org/abs/2304.03442) (Park et al., 2023) | Memory stream + recency/importance/relevance retrieval + reflection |
| 8 | `08_tree_of_thoughts.py` | [Tree of Thoughts](https://arxiv.org/abs/2305.10601) (Yao et al., 2023) | Search over partial solutions: propose, evaluate, keep the best |
| 9 | `09_voyager.py` | [Voyager](https://arxiv.org/abs/2305.16291) (Wang et al., 2023) | Curriculum + code-writing with feedback + a growing skill library |
| 10 | `10_autogen.py` | [AutoGen](https://arxiv.org/abs/2308.08155) (Wu et al., 2023) | Agents that solve tasks by talking to each other (LLM + code executor) |

## Running

```bash
python3 03_react.py           # offline: scripted LLM replies, no API key needed
python3 03_react.py --live    # real model (Claude), see below
```

**Offline mode (default).** A `ScriptedLLM` in `common.py` replays canned replies,
so every run gives the same output and you can trace the mechanism step by step.
The agent logic (parsing, tool calls, memory, search, retries) is real; only the
model's text is pre-written.

**Live mode (`--live`).** Uses `claude-opus-5-5` through the Anthropic SDK:

```bash
pip install anthropic
export ANTHROPIC_API_KEY=...   # or: ant auth login
```

In live mode, Reflexion, Voyager and AutoGen **execute code the model writes** on
your machine (AutoGen in a subprocess, the others with `exec`). They're
teaching demos, not sandboxes.

## How the papers build on each other

```
Chain-of-Thought ──► ReAct ◄── MRKL / Toolformer   (reasoning + tools)
                      │
          ┌───────────┼──────────────┐
      Reflexion   Self-Refine   Tree of Thoughts   (self-improvement & search)
          │
   Generative Agents, Voyager                      (memory & lifelong skills)
          │
       AutoGen                                     (multiple agents)
```
