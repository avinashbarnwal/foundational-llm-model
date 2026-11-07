"""Paper 2: MRKL Systems - A Modular, Neuro-Symbolic Architecture
Karpas et al. (AI21 Labs), 2022 - https://arxiv.org/abs/2205.00445

Core idea
---------
An LLM alone is bad at exact math, fresh facts, and private data. MRKL
("miracle") keeps the LLM as a *router* and sends each query to the expert
module best suited for it: a calculator, a database, an API, or the LLM itself.

    query -> [LLM router] -> pick expert + extract its input -> expert -> answer

This is the ancestor of "tool use" / function calling in every agent framework.
"""
from common import banner, calculate, get_llm

# --- Expert modules: deterministic, reliable, narrow -------------------------
FX_RATES = {"USD": 1.0, "EUR": 0.92, "INR": 83.2}
WEATHER_DB = {"paris": "14°C, light rain", "mumbai": "31°C, humid", "seattle": "9°C, cloudy"}


def currency_expert(arg):
    amount, src, dst = arg.split()
    return f"{float(amount) / FX_RATES[src] * FX_RATES[dst]:.2f} {dst}"


def weather_expert(city):
    return WEATHER_DB.get(city.strip().lower(), "unknown city")


EXPERTS = {
    "calculator": (calculate, "arithmetic expression, e.g. 17 * 23"),
    "currency": (currency_expert, "'<amount> <FROM> <TO>', e.g. 100 USD INR"),
    "weather": (weather_expert, "city name"),
    "llm": (None, "anything else; answer it yourself"),
}

ROUTER_PROMPT = """You route a user query to exactly one expert.
Experts:
{experts}

Reply with one line in the form  expert_name: input
Query: {query}"""

QUERIES = [
    "What is 1234 * 5678?",
    "Convert 250 dollars to rupees",
    "How's the weather in Paris?",
    "Who wrote Pride and Prejudice?",
]

SCRIPTED = [
    "calculator: 1234 * 5678",
    "currency: 250 USD INR",
    "weather: Paris",
    "llm: Jane Austen wrote Pride and Prejudice (1813).",
]


def mrkl(query, llm):
    experts = "\n".join(f"- {name}: {desc}" for name, (_, desc) in EXPERTS.items())
    decision = llm(ROUTER_PROMPT.format(experts=experts, query=query)).strip()
    name, _, arg = decision.partition(":")
    name, arg = name.strip(), arg.strip()
    print(f"router -> {name}({arg!r})")
    fn = EXPERTS.get(name, (None,))[0]
    return arg if fn is None else fn(arg)


def main():
    llm = get_llm(SCRIPTED)
    for q in QUERIES:
        banner(q)
        print("answer:", mrkl(q, llm))


if __name__ == "__main__":
    main()
