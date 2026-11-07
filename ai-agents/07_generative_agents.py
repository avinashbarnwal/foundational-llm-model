"""Paper 7: Generative Agents - Interactive Simulacra of Human Behavior
Park et al. (Stanford/Google), 2023 - https://arxiv.org/abs/2304.03442

Core idea
---------
Believable long-lived agents need memory architecture, not just a big prompt:

1. Memory stream - every observation is stored with a timestamp and an
   importance score (1-10, rated by the LLM).
2. Retrieval - for a query, score each memory as
       recency (exponential decay) + importance + relevance (similarity)
   and put only the top-k into the prompt.
3. Reflection - when recent importance adds up past a threshold, ask the LLM
   for higher-level insights and store them back as new memories.

The paper uses embeddings for relevance; here word overlap keeps it dependency-free.
"""
import re
from dataclasses import dataclass

from common import banner, get_llm


@dataclass
class Memory:
    text: str
    hour: float  # simulated time the memory was created / last accessed
    importance: int


def words(text):
    return set(re.findall(r"[a-z]+", text.lower())) - {"the", "a", "is", "to", "and", "of", "at", "for", "in", "her", "she"}


class MemoryStream:
    def __init__(self, llm, reflect_threshold=25):
        self.llm = llm
        self.memories = []
        self.reflect_threshold = reflect_threshold
        self.importance_since_reflection = 0

    def add(self, text, hour):
        reply = self.llm(
            "On a scale of 1 (mundane, e.g. brushing teeth) to 10 (poignant, e.g. a breakup), "
            f"rate the importance of this memory. Reply with a number only.\nMemory: {text}"
        )
        importance = int(re.search(r"\d+", reply).group())
        self.memories.append(Memory(text, hour, importance))
        self.importance_since_reflection += importance
        print(f"  [t={hour:>4}h imp={importance:>2}] {text}")
        if self.importance_since_reflection >= self.reflect_threshold:
            self.reflect(hour)

    def retrieve(self, query, now, k=3, decay=0.99):
        q = words(query)

        def score(m):
            recency = decay ** (now - m.hour)
            importance = m.importance / 10
            relevance = len(q & words(m.text)) / max(len(q), 1)
            return recency + importance + relevance

        top = sorted(self.memories, key=score, reverse=True)[:k]
        for m in top:
            m.hour = now  # accessing a memory refreshes its recency
        return top

    def reflect(self, now):
        recent = "\n".join(m.text for m in self.memories[-5:])
        insight = self.llm(
            f"Statements about Isabella:\n{recent}\n\n"
            "What is one high-level insight you can infer from these statements? One sentence."
        ).strip()
        print(f"  ** reflection triggered -> {insight}")
        self.importance_since_reflection = 0
        self.memories.append(Memory(insight, now, 8))


SCRIPTED = [
    "2", "3", "8", "7", "6",  # importance of the five observations
    "Isabella is the organiser of a Valentine's Day party and is actively recruiting guests.",
    "1",  # importance of the final observation
    "Tonight Isabella should finish decorating Hobbs Cafe and remind Maria and Klaus about the party.",
]


def main():
    llm = get_llm(SCRIPTED)
    stream = MemoryStream(llm)

    banner("Observations flow into the memory stream")
    stream.add("Isabella ate breakfast at home", hour=1)
    stream.add("Isabella opened Hobbs Cafe for the day", hour=2)
    stream.add("Isabella decided to host a Valentine's Day party at Hobbs Cafe", hour=3)
    stream.add("Isabella invited Maria to the Valentine's Day party", hour=5)
    stream.add("Maria agreed to help decorate the cafe for the party", hour=6)
    stream.add("Isabella wiped the counter", hour=9)

    banner("Retrieval for: 'What should Isabella do tonight about the party?'")
    query = "What should Isabella do tonight about the party?"
    top = stream.retrieve(query, now=10)
    for m in top:
        print("  -", m.text)

    plan = llm("Relevant memories:\n" + "\n".join(m.text for m in top) + f"\n\n{query} Answer in one sentence.")
    print("\nPlan:", plan)


if __name__ == "__main__":
    main()
