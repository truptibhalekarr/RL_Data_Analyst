"""
inference.py — OpenEnv-compatible HTTP API server
Exposes POST /reset and POST /step endpoints as required by the hackathon checker.
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import os
import sys

# Make sure local modules are importable
sys.path.insert(0, os.path.dirname(__file__))

from env import DataAnalystEnv
from agent import make_agent

app = FastAPI(
    title="Data Analyst RL Environment",
    description="Reinforcement Learning environment for automated data analysis",
    version="1.0.0"
)

# ---------------------------------------------------------------------------
# Global state — one env + agent instance shared across requests
# ---------------------------------------------------------------------------
CSV_PATH = os.path.join(os.path.dirname(__file__), "dataset.csv")
env = DataAnalystEnv(CSV_PATH)
agent = make_agent("heuristic")
current_state = None


# ---------------------------------------------------------------------------
# GET / — health check
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "status": "ok",
        "project": "Data Analyst RL Environment",
        "description": "RL agent that learns automated data analysis",
        "endpoints": ["/reset", "/step", "/validate", "/health"]
    }


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "healthy"}


# ---------------------------------------------------------------------------
# POST /reset  ← THIS is what the hackathon checker hits
# ---------------------------------------------------------------------------
@app.post("/reset")
def reset():
    global current_state, agent
    agent.reset()
    current_state = env.reset()

    return {
        "status": "ok",
        "message": "Environment reset successfully",
        "state": current_state.to_dict(),
        "observation": {
            "step": current_state.step,
            "task_id": current_state.task.task_id,
            "task_name": current_state.task.name,
            "task_description": current_state.task.description,
            "dataset_shape": current_state.dataset_summary["shape"],
            "columns": current_state.dataset_summary["columns"],
        }
    }


# ---------------------------------------------------------------------------
# POST /step
# ---------------------------------------------------------------------------
@app.post("/step")
async def step(request: Request):
    global current_state

    if current_state is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Environment not reset. Call POST /reset first."}
        )

    if current_state.done:
        return JSONResponse(
            status_code=400,
            content={"error": "Episode is done. Call POST /reset to start a new one."}
        )

    # Get action from request body, or let agent decide
    try:
        body = await request.json()
        action = body.get("action", None)
    except Exception:
        action = None

    # If no action provided, use the heuristic agent
    if action is None:
        action = agent.select_action(current_state)

    next_state, reward, done, info = env.step(action)
    agent.update(reward, done)
    current_state = next_state

    result = info["grade_result"]

    return {
        "status": "ok",
        "reward": reward,
        "done": done,
        "step": result.task_id,
        "task_name": result.task_name,
        "score_pct": result.score_pct,
        "passed": result.passed,
        "feedback": result.feedback,
        "breakdown": result.breakdown,
        "next_state": next_state.to_dict() if not done else None
    }


# ---------------------------------------------------------------------------
# POST /validate — run full episode and return summary
# ---------------------------------------------------------------------------
@app.post("/validate")
def validate():
    local_env = DataAnalystEnv(CSV_PATH)
    local_agent = make_agent("heuristic")
    state = local_env.reset()
    local_agent.reset()

    results = []
    while not state.done:
        action = local_agent.select_action(state)
        state, reward, done, info = local_env.step(action)
        local_agent.update(reward, done)
        results.append(info["grade_result"])

    from grader import total_score
    total, max_r, pct = total_score(results)

    return {
        "status": "ok",
        "total_reward": total,
        "max_reward": max_r,
        "score_pct": pct,
        "passed": pct >= 50.0,
        "steps": [
            {
                "task_id": r.task_id,
                "task_name": r.task_name,
                "reward": r.reward,
                "score_pct": r.score_pct,
                "passed": r.passed,
                "feedback": r.feedback,
            }
            for r in results
        ]
    }


# ---------------------------------------------------------------------------
# GET /env-info — environment metadata
# ---------------------------------------------------------------------------
@app.get("/env-info")
def env_info():
    return {
        "name": "DataAnalystEnv",
        "version": "1.0.0",
        "tasks": [
            {"id": 0, "name": "Column Classification"},
            {"id": 1, "name": "KPI Selection"},
            {"id": 2, "name": "Chart Selection"},
            {"id": 3, "name": "Insight Generation"},
        ],
        "agents_available": ["random", "rule", "heuristic"],
        "dataset": "Sales dataset (30 rows, 9 columns)",
        "reward_range": [-10, 30.5]
    }


# ---------------------------------------------------------------------------
# Run server
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("inference:app", host="0.0.0.0", port=port, reload=False)