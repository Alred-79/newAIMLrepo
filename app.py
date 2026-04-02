"""
ClinEQ MVP1 Demo App
Run: python app.py
Open: http://localhost:8000
"""

import sys
from pathlib import Path

# Ensure ai-engine packages are importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent / "ai-engine"))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from patient_sim.scenario_loader import load_and_seed_scenario
from shared.session import SessionManager
from shared.models import ScenarioConfig
from shared.config import Config

app = FastAPI(title="ClinEQ MVP1 Demo")

# Load all scenario JSONs at startup
SCENARIOS: dict[str, ScenarioConfig] = {}
SCENARIO_INDICES: dict[str, object] = {}  # ScenarioIndex objects for BM25 hybrid retrieval
SCENARIOS_DIR = Path("ai-engine/data/scenarios")


@app.on_event("startup")
async def startup():
    """Load and seed all scenarios on server start."""
    if not SCENARIOS_DIR.exists():
        print(f"WARNING: {SCENARIOS_DIR} does not exist. No scenarios loaded.")
        return

    for json_file in sorted(SCENARIOS_DIR.glob("*.json")):
        try:
            scenario, index = await load_and_seed_scenario(json_file)
            SCENARIOS[scenario.scenario_id] = scenario
            SCENARIO_INDICES[scenario.scenario_id] = index
            print(f"  Loaded: {scenario.scenario_id} ({len(scenario.content_nodes)} nodes)")
        except Exception as e:
            print(f"  FAILED to load {json_file.name}: {e}")

    # Warm up the models so the first user request isn't slow
    print("Warming up Ollama models...")
    try:
        from shared.llm import llm_completion, llm_embedding
        await llm_completion(
            model=Config.LIGHTWEIGHT_MODEL,
            system_prompt="You are a test.",
            user_prompt="Respond with: ok",
            max_tokens=5,
        )
        await llm_embedding(["warmup test"])
        print("Models warmed up.")
    except Exception as e:
        print(f"Warmup failed (non-fatal): {e}")

    print(f"\n{len(SCENARIOS)} scenarios ready.\n")



@app.get("/api/scenarios")
async def list_scenarios():
    """Return available scenarios for the frontend dropdown."""
    # Control scenario should appear first
    scenarios = []
    for _, s in SCENARIOS.items():
        scenarios.append({
            "scenario_id": s.scenario_id,
            "title": s.title,
            "category": s.category.value,
            "bias_direction": s.bias_direction.value,
            "node_count": len(s.content_nodes),
        })

    # Sort: control first, then alphabetical
    scenarios.sort(key=lambda x: (0 if x["category"] == "control" else 1, x["title"]))
    return {"scenarios": scenarios}


# Active sessions keyed by websocket connection
ACTIVE_SESSIONS: dict[int, SessionManager] = {}
# Control encounter data, keyed by websocket connection ID
# Stores profile + history after control scenario ends for delta analysis
CONTROL_DATA: dict[int, dict] = {}


@app.websocket("/ws/chat/{scenario_id}")
async def chat_websocket(websocket: WebSocket, scenario_id: str):
    """
    WebSocket endpoint for a single chat session.

    Client sends: {"type": "message", "content": "student text"}
                  {"type": "end_encounter"}
    Server sends: {"type": "messages", "messages": [...], "metadata": {...}}
                  {"type": "encounter_ended", "summary": {...}}
                  {"type": "error", "message": "..."}
    """
    await websocket.accept()

    if scenario_id not in SCENARIOS:
        await websocket.send_json({
            "type": "error",
            "message": f"Scenario '{scenario_id}' not found.",
        })
        await websocket.close()
        return

    scenario = SCENARIOS[scenario_id]
    scenario_index = SCENARIO_INDICES.get(scenario_id)
    ws_id = id(websocket)

    # Pass control data to experimental sessions for delta analysis
    control_info = CONTROL_DATA.get(ws_id, {})
    session = SessionManager(
        scenario,
        scenario_index=scenario_index,
        control_profile=control_info.get("profile"),
        control_history=control_info.get("history"),
        control_scenario=control_info.get("scenario"),
    )

    # Send the opening line immediately
    # The patient's opening line is delivered when the student first speaks,
    # so we just send a welcome system message here
    await websocket.send_json({
        "type": "messages",
        "messages": [
            {
                "role": "system",
                "content": f"Encounter started with {scenario.patient_profile.name}. "
                           f"You are a medical student conducting a clinical interview.",
            },
        ],
        "metadata": {"exchange_count": 0},
    })

    try:
        while True:
            # Receive student message
            data = await websocket.receive_json()
            msg_type = data.get("type", "")

            if msg_type == "end_encounter":
                end_result = await session.end_encounter()
                # Store control data for future experimental sessions
                if end_result.get("control_profile"):
                    CONTROL_DATA[ws_id] = {
                        "profile": end_result["control_profile"],
                        "history": list(session.state.conversation_history),
                        "scenario": scenario,
                    }
                await websocket.send_json({
                    "type": "encounter_ended",
                    "messages": end_result["messages"],
                    "summary": end_result["metadata"],
                })
                break

            elif msg_type == "message":
                content = data.get("content", "").strip()
                if not content:
                    continue

                # Process the student message through the full pipeline
                try:
                    result = await session.process_student_message(content)

                    await websocket.send_json({
                        "type": "messages",
                        "messages": result["messages"],
                        "metadata": result["metadata"],
                        "is_complete": result["is_complete"],
                    })

                    if result["is_complete"]:
                        # Auto-end if max exchanges reached
                        end_result = await session.end_encounter()
                        if end_result.get("control_profile"):
                            CONTROL_DATA[ws_id] = {
                                "profile": end_result["control_profile"],
                                "history": list(session.state.conversation_history),
                                "scenario": scenario,
                            }
                        await websocket.send_json({
                            "type": "encounter_ended",
                            "messages": end_result["messages"],
                            "summary": end_result["metadata"],
                        })
                        break

                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Processing error: {str(e)}",
                    })

    except WebSocketDisconnect:
        pass



@app.get("/")
async def serve_frontend():
    """Serve the single-page chat frontend."""
    html_path = Path("demo-frontend/index.html")
    if html_path.exists():
        return HTMLResponse(html_path.read_text())
    else:
        return HTMLResponse(
            "<h1>ClinEQ MVP1</h1>"
            "<p>demo-frontend/index.html not found. "
            "Create it per Step 4 instructions.</p>"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
