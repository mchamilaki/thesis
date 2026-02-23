import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import csv
from langchain_core.messages import HumanMessage
from main import router_node
from src.state import AgentState

correct = 0
total = 0

with open("evaluation/intent_test_set.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    print("Detected columns:", reader.fieldnames)

    for row in reader:
        prompt = row["prompt"]
        expected = row["expected_route"]

        # Minimal state required by router_node
        state: AgentState = {
            "messages": [HumanMessage(content=prompt)],
            "current_flow": None,
        }

        result = router_node(state)

        predicted = result.get("route_to")         
                  

        if predicted != expected:
            print(f"Prompt: {prompt}")
            print(f"Expected: {expected} | Predicted: {predicted}")
            print("-" * 50)

        if predicted == expected:
            correct += 1

        total += 1
        



accuracy = correct / total
print(f"\nIntent Accuracy: {accuracy:.2%}")