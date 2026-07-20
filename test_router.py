from main import router_llm, ROUTER_SYSTEM
from langchain_core.messages import SystemMessage, HumanMessage

# --- Part 1: single-message classification ---
test_messages = [
    ("how much do I owe", "billing"),
    ("my internet is down", "technical_support"),
    ("hello!", "chitchat"),
    ("what fiber plans do you have", "info_lookup"),
    ("this is unacceptable, I want a person", "escalation"),
    ("asdfgh", "unknown"),
]

print("=== Single-message classification ===")
for msg, expected in test_messages:
    r = router_llm.invoke([
        SystemMessage(content=ROUTER_SYSTEM),
        HumanMessage(content=f"Conversation:\nUser: {msg}\n\nClassify the latest user message."),
    ])
    status = "OK " if r.intent == expected else "MISS"
    print(f"[{status}] {msg!r:42} -> {r.intent:20} ({r.confidence:.2f})  expected: {expected}")

# --- Part 2: contextual follow-up (the critical test for removing current_flow) ---
print("\n=== Contextual follow-up: '123' after billing question ===")
for i in range(3):  # run a few times to check consistency
    r = router_llm.invoke([
        SystemMessage(content=ROUTER_SYSTEM),
        HumanMessage(content=(
            "Conversation:\n"
            "User: How much do I owe?\n"
            "Assistant: Could you please provide your account ID so I can retrieve your billing details?\n"
            "User: 123\n\n"
            "Classify the latest user message."
        )),
    ])
    status = "OK " if r.intent == "billing" else "MISS"
    print(f"[{status}] run {i+1}: '123' -> {r.intent} ({r.confidence:.2f})  expected: billing")