"""
inference.py — OpenEnv-compatible FastAPI server
Required by the hackathon evaluator at POST /reset
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import DataAnalystEnv
from agent import make_agent
from grader import total_score

app = FastAPI(
    title="Data Analyst RL Environment",
    description="Reinforcement Learning environment for automated data analysis",
    version="1.0.0"
)

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.csv")
if not os.path.exists(CSV_PATH):
    CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dataset.csv")

env = DataAnalystEnv(CSV_PATH)
agent = make_agent("heuristic")
current_state = None


@app.get("/")
def root():
    return {"status": "ok", "project": "Data Analyst RL Environment"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/reset")
def reset():
    global current_state, env, agent
    try:
        agent.reset()
        current_state = env.reset()
        return JSONResponse(status_code=200, content={
            "status": "ok",
            "message": "Environment reset successfully",
            "observation": {
                "step": current_state.step,
                "task_id": current_state.task.task_id,
                "task_name": current_state.task.name,
                "task_description": current_state.task.description,
                "dataset_shape": current_state.dataset_summary["shape"],
                "columns": current_state.dataset_summary["columns"],
                "done": current_state.done,
                "cumulative_reward": 0.0
            }
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.post("/step")
async def step(request: Request):
    global current_state
    if current_state is None:
        return JSONResponse(status_code=400, content={"error": "Call POST /reset first."})
    if current_state.done:
        return JSONResponse(status_code=400, content={"error": "Episode done. Call POST /reset."})
    try:
        body = await request.json()
        action = body.get("action", None)
    except Exception:
        action = None
    if action is None:
        action = agent.select_action(current_state)
    next_state, reward, done, info = env.step(action)
    agent.update(reward, done)
    current_state = next_state
    result = info["grade_result"]
    return JSONResponse(status_code=200, content={
        "status": "ok", "reward": reward, "done": done,
        "task_name": result.task_name, "score_pct": result.score_pct,
        "passed": result.passed, "feedback": result.feedback,
    })


@app.post("/validate")
def validate():
    try:
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
        total, max_r, pct = total_score(results)
        return JSONResponse(status_code=200, content={
            "status": "ok", "total_reward": total, "max_reward": max_r,
            "score_pct": pct, "passed": pct >= 50.0,
            "steps": [{"task_id": r.task_id, "task_name": r.task_name,
                       "reward": r.reward, "score_pct": r.score_pct,
                       "passed": r.passed} for r in results]
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.get("/env-info")
def env_info():
    return {"name": "DataAnalystEnv", "version": "1.0.0",
            "tasks": [{"id": 0, "name": "Column Classification"},
                      {"id": 1, "name": "KPI Selection"},
                      {"id": 2, "name": "Chart Selection"},
                      {"id": 3, "name": "Insight Generation"}]}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("inference:app", host="0.0.0.0", port=port, reload=False)