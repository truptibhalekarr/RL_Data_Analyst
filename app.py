import gradio as gr
import pandas as pd
import json
from env import DataAnalystEnv
from agent import make_agent
from grader import total_score

def run_agent(agent_type: str):
    env = DataAnalystEnv("dataset.csv")
    agent = make_agent(agent_type)
    state = env.reset()
    agent.reset()

    log = []
    results = []

    while not state.done:
        action = agent.select_action(state)
        state, reward, done, info = env.step(action)
        agent.update(reward, done)
        results.append(info["grade_result"])

        result = info["grade_result"]
        status = "✅ PASSED" if result.passed else "❌ FAILED"
        log.append(
            f"**[Step {result.task_id}] {result.task_name}** — {status}\n"
            f"- Reward: `{result.reward:+.2f}` / `{result.max_possible_reward:.2f}`\n"
            f"- Score: `{result.score_pct:.1f}%`\n"
            f"- Feedback: {result.feedback}\n"
        )

    total, max_r, pct = total_score(results)
    overall = f"## 🏆 TOTAL: {total:.2f} / {max_r:.2f} ({pct:.1f}%) — {'✅ PASS' if pct >= 50 else '❌ FAIL'}"

    return "\n---\n".join(log), overall


with gr.Blocks(title="Data Analyst RL Environment") as demo:
    gr.Markdown("# 🤖 Data Analyst Reinforcement Learning Environment")
    gr.Markdown(
        "An RL agent learns to analyze data step by step: "
        "**Column Classification → KPI Selection → Chart Selection → Insight Generation**"
    )

    with gr.Row():
        agent_choice = gr.Dropdown(
            choices=["random", "rule", "heuristic"],
            value="heuristic",
            label="Select Agent"
        )
        run_btn = gr.Button("▶ Run Episode", variant="primary")

    step_output = gr.Markdown(label="Step-by-Step Results")
    total_output = gr.Markdown(label="Final Score")

    run_btn.click(
        fn=run_agent,
        inputs=[agent_choice],
        outputs=[step_output, total_output]
    )

demo.launch()