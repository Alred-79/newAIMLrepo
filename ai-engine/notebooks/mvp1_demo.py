"""
ClinEQ MVP1 Demo — Full Pipeline Walkthrough
Run this after docker compose up -d and pip install -e ai-engine/
"""
import asyncio
from patient_sim.scenario_loader import load_and_seed_scenario
from shared.session import SessionManager


async def main():
    # 1. Load and seed the control scenario
    scenario = await load_and_seed_scenario(
        "ai-engine/data/scenarios/control_john_smith.json"
    )
    print(f"Loaded scenario: {scenario.title}")
    print(f"Content nodes: {len(scenario.content_nodes)}")
    print(f"Opening line: {scenario.opening_line}")
    print("---")

    # 2. Create a session
    session = SessionManager(scenario)

    # 3. Simulate a conversation
    test_messages = [
        "Hello, how are you today?",
        "What brings you in today?",
        "When did the rash start?",
        "Does anything make it better or worse?",
        "Do you have any allergies?",
        "Are you taking any medications?",
        "Has anyone in your family had similar issues?",
    ]

    for msg in test_messages:
        print(f"\n[Student]: {msg}")
        result = await session.process_student_message(msg)

        for m in result["messages"]:
            if m["role"] == "system":
                print(f"[System]: {m['content']}")
            elif m["role"] == "patient":
                print(f"[Patient]: {m['content']}")

        meta = result["metadata"]
        print(f"  (method: {meta['match_method']}, "
              f"exchange: {meta['exchange_count']}, "
              f"nodes: {meta.get('nodes_delivered', [])})")

        if result["is_complete"]:
            print("\n--- Encounter complete ---")
            break

    # 4. End the encounter
    end_result = session.end_encounter()
    print(f"\nSession summary:")
    print(f"  Exchanges: {end_result['metadata']['exchange_count']}")
    print(f"  Nodes delivered: {end_result['metadata']['total_nodes_delivered']}"
          f" / {end_result['metadata']['total_nodes_available']}")
    if end_result['metadata']['undelivered_essential_nodes']:
        print(f"  Undelivered essential: "
              f"{end_result['metadata']['undelivered_essential_nodes']}")


asyncio.run(main())
