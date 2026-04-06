"""
env.py — Data Analyst Reinforcement Learning Environment

Follows the standard RL loop:
    state  →  agent takes action  →  env returns (next_state, reward, done, info)

Steps in the episode:
    0. Column Classification
    1. KPI Selection
    2. Chart Selection
    3. Insight Generation
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass, field

from tasks import ALL_TASKS, Task, get_task
from grader import grade, GradeResult, total_score


# ---------------------------------------------------------------------------
# State representation
# ---------------------------------------------------------------------------

@dataclass
class State:
    """Snapshot of the environment at a given step."""
    step: int                              # current task index (0–3)
    task: Task                             # current Task object
    dataset_summary: Dict[str, Any]       # lightweight dataset metadata
    history: list = field(default_factory=list)  # list of past GradeResults
    done: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "task_id": self.task.task_id,
            "task_name": self.task.name,
            "task_description": self.task.description,
            "task_hint": self.task.hint,
            "valid_actions_sample": self.task.valid_actions[:1] if self.task.valid_actions else [],
            "dataset_summary": self.dataset_summary,
            "completed_steps": [r.task_name for r in self.history],
            "cumulative_reward": round(sum(r.reward for r in self.history), 2),
            "done": self.done,
        }

    def __str__(self) -> str:
        lines = [
            f"{'─'*60}",
            f"  STATE  |  Step {self.step}  →  {self.task.name}",
            f"{'─'*60}",
            f"  Task        : {self.task.description}",
            f"  Hint        : {self.task.hint}",
            f"  Done        : {self.done}",
            f"  Cum Reward  : {sum(r.reward for r in self.history):.2f}",
        ]
        if self.history:
            lines.append(f"  Past tasks  : {[r.task_name for r in self.history]}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class DataAnalystEnv:
    """
    RL Environment for automated data analysis.

    Usage
    -----
    env = DataAnalystEnv("dataset.csv")
    state = env.reset()

    while not state.done:
        action = agent.select_action(state)
        state, reward, done, info = env.step(action)
    """

    PASS_THRESHOLD = 50.0   # minimum score% to consider a step "passed"

    def __init__(self, csv_path: str = "dataset.csv"):
        self.csv_path = csv_path
        self.df: Optional[pd.DataFrame] = None
        self._current_step: int = 0
        self._history: list = []
        self._episode_rewards: list = []
        self._load_dataset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> State:
        """Reset the environment and return the initial state."""
        self._current_step = 0
        self._history = []
        self._episode_rewards = []
        return self._make_state()

    def step(self, action: Any) -> Tuple[State, float, bool, Dict[str, Any]]:
        """
        Execute one step.

        Parameters
        ----------
        action : Any
            The agent's action for the current task.

        Returns
        -------
        next_state : State
        reward     : float
        done       : bool
        info       : dict  (grading details, feedback, etc.)
        """
        if self._is_done():
            raise RuntimeError("Episode is done. Call env.reset() to start a new one.")

        task = get_task(self._current_step)
        result: GradeResult = grade(action, task)

        self._history.append(result)
        self._episode_rewards.append(result.reward)

        self._current_step += 1
        done = self._is_done()

        next_state = self._make_state(done=done)

        info = {
            "grade_result": result,
            "reward": result.reward,
            "score_pct": result.score_pct,
            "passed": result.passed,
            "feedback": result.feedback,
            "breakdown": result.breakdown,
        }

        return next_state, result.reward, done, info

    def render(self, state: Optional[State] = None) -> None:
        """Print current environment state to stdout."""
        s = state or self._make_state()
        print(s)

    def summary(self) -> str:
        """Return a full episode summary after all steps complete."""
        if not self._history:
            return "No steps completed yet."

        total, max_r, pct = total_score(self._history)
        lines = [
            "",
            "╔" + "═" * 58 + "╗",
            "║      DATA ANALYST RL ENVIRONMENT — EPISODE SUMMARY      ║",
            "╚" + "═" * 58 + "╝",
        ]
        for r in self._history:
            status = "✅" if r.passed else "❌"
            lines.append(
                f"  {status} [{r.task_id}] {r.task_name:<25} "
                f"Reward: {r.reward:+6.2f}  ({r.score_pct:.1f}%)"
            )
        lines += [
            "  " + "─" * 56,
            f"  🏆 TOTAL  :  {total:+.2f} / {max_r:.2f}  ({pct:.1f}%)",
            f"  {'PASS ✅' if pct >= self.PASS_THRESHOLD else 'FAIL ❌'}",
            "═" * 60,
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Dataset utilities (exposed so agent can observe the data)
    # ------------------------------------------------------------------

    @property
    def dataframe(self) -> pd.DataFrame:
        return self.df.copy()

    def get_dataset_summary(self) -> Dict[str, Any]:
        return self._build_dataset_summary()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_dataset(self) -> None:
        try:
            self.df = pd.read_csv(self.csv_path, parse_dates=["date"])
        except Exception as e:
            raise FileNotFoundError(f"Could not load dataset: {e}")

    def _is_done(self) -> bool:
        return self._current_step >= len(ALL_TASKS)

    def _make_state(self, done: bool = False) -> State:
        if self._is_done() and not done:
            done = True
        task = get_task(min(self._current_step, len(ALL_TASKS) - 1))
        return State(
            step=self._current_step,
            task=task,
            dataset_summary=self._build_dataset_summary(),
            history=list(self._history),
            done=done,
        )

    def _build_dataset_summary(self) -> Dict[str, Any]:
        df = self.df
        col_types = {}
        for col in df.columns:
            dtype = str(df[col].dtype)
            if "datetime" in dtype:
                col_types[col] = "datetime"
            elif dtype in ("object", "bool", "category") or df[col].dtype == "O":
                col_types[col] = "categorical"
            elif pd.api.types.is_numeric_dtype(df[col]):
                col_types[col] = "numerical"
            else:
                col_types[col] = "categorical"

        numerical_cols = [c for c, t in col_types.items() if t == "numerical"]
        num_stats = {}
        if numerical_cols:
            stats_df = df[numerical_cols].describe().round(2)
            num_stats = stats_df.to_dict()

        cat_cols = [c for c, t in col_types.items() if t == "categorical"]
        cat_info = {col: df[col].value_counts().head(5).to_dict() for col in cat_cols}

        return {
            "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
            "columns": list(df.columns),
            "column_types_inferred": col_types,
            "numerical_stats": num_stats,
            "categorical_top_values": cat_info,
            "missing_values": df.isnull().sum().to_dict(),
            "date_range": {
                "min": str(df["date"].min().date()) if "date" in df.columns else None,
                "max": str(df["date"].max().date()) if "date" in df.columns else None,
            },
        }


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    env = DataAnalystEnv("dataset.csv")
    state = env.reset()
    print(state)
    print("\nDataset Summary:")
    import json
    print(json.dumps(env.get_dataset_summary(), indent=2))