"""
grader.py — Reward / Grading System for the Data Analyst RL Environment.

Each grader function receives the agent's action and the Task object,
computes a reward, and returns a GradeResult with detailed feedback.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
import re

from tasks import (
    Task,
    COLUMN_CLASSIFICATION_TASK,
    KPI_SELECTION_TASK,
    CHART_SELECTION_TASK,
    INSIGHT_GENERATION_TASK,
)


# ---------------------------------------------------------------------------
# Data class for grading result
# ---------------------------------------------------------------------------

@dataclass
class GradeResult:
    task_id: int
    task_name: str
    reward: float
    max_possible_reward: float
    score_pct: float          # 0–100
    passed: bool              # True if agent cleared minimum threshold
    feedback: str
    breakdown: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        return (
            f"\n{'─'*55}\n"
            f"[Task {self.task_id}] {self.task_name}  —  {status}\n"
            f"  Reward      : {self.reward:+.2f}  /  {self.max_possible_reward:.2f}\n"
            f"  Score       : {self.score_pct:.1f}%\n"
            f"  Feedback    : {self.feedback}\n"
            f"  Breakdown   : {self.breakdown}\n"
            f"{'─'*55}"
        )


# ---------------------------------------------------------------------------
# GRADER 0 — Column Classification
# ---------------------------------------------------------------------------

def grade_column_classification(action: Dict[str, str], task: Task) -> GradeResult:
    """
    action: {column_name: predicted_type, ...}
    Scores per-column correctness; gives bonus for perfect match.
    """
    cfg = task.reward_config
    expected = task.expected_output
    max_reward = cfg["perfect_match"] + len(expected) * cfg["per_correct"]

    if not isinstance(action, dict):
        return GradeResult(
            task_id=task.task_id, task_name=task.name,
            reward=-1.0, max_possible_reward=max_reward,
            score_pct=0.0, passed=False,
            feedback="Action must be a dict mapping column names to types.",
        )

    correct, wrong, missing = [], [], []
    reward = 0.0

    for col, exp_type in expected.items():
        pred = action.get(col)
        if pred is None:
            missing.append(col)
        elif pred == exp_type:
            correct.append(col)
            reward += cfg["per_correct"]
        else:
            wrong.append(f"{col}(got={pred},exp={exp_type})")
            reward += cfg["per_wrong"]

    fraction_correct = len(correct) / len(expected)

    if fraction_correct == 1.0:
        reward += cfg["perfect_match"]
        feedback = "Perfect classification! All columns correctly typed."
        passed = True
    elif fraction_correct >= cfg["partial_threshold"]:
        reward += cfg["partial_reward"]
        feedback = f"Good — {len(correct)}/{len(expected)} correct. Partial bonus awarded."
        passed = True
    else:
        feedback = f"Only {len(correct)}/{len(expected)} correct. Needs improvement."
        passed = False

    reward = max(reward, -len(expected) * abs(cfg["per_wrong"]))
    score_pct = max(0.0, (reward / max_reward) * 100)

    return GradeResult(
        task_id=task.task_id, task_name=task.name,
        reward=round(reward, 2), max_possible_reward=max_reward,
        score_pct=round(score_pct, 1), passed=passed,
        feedback=feedback,
        breakdown={
            "correct": correct, "wrong": wrong,
            "missing": missing, "fraction_correct": round(fraction_correct, 2),
        },
    )


# ---------------------------------------------------------------------------
# GRADER 1 — KPI Selection
# ---------------------------------------------------------------------------

def grade_kpi_selection(action: List[str], task: Task) -> GradeResult:
    """
    action: list of KPI strings the agent selected.
    """
    cfg = task.reward_config
    expected = task.expected_output
    required = set(expected["required"])
    bonus    = set(expected["bonus"])
    penalised = set(expected["penalised"])

    max_reward = (
        len(required) * cfg["per_required"]
        + len(bonus)  * cfg["per_bonus"]
        + cfg["perfect_reward"]
    )

    if not isinstance(action, list):
        return GradeResult(
            task_id=task.task_id, task_name=task.name,
            reward=-1.0, max_possible_reward=max_reward,
            score_pct=0.0, passed=False,
            feedback="Action must be a list of KPI strings.",
        )

    selected = set(action)
    got_required  = selected & required
    got_bonus     = selected & bonus
    got_penalised = selected & penalised

    reward = 0.0
    reward += len(got_required)  * cfg["per_required"]
    reward += len(got_bonus)     * cfg["per_bonus"]
    reward += len(got_penalised) * cfg["per_penalised"]

    if required.issubset(selected) and not got_penalised:
        reward += cfg["perfect_reward"]
        feedback = "All required KPIs selected with no penalised ones!"
        passed = True
    elif len(got_required) >= len(required) * 0.6:
        feedback = f"Covered {len(got_required)}/{len(required)} required KPIs."
        passed = True
    else:
        feedback = f"Too few required KPIs. Got {len(got_required)}/{len(required)}."
        passed = False

    score_pct = max(0.0, min(100.0, (reward / max_reward) * 100))

    return GradeResult(
        task_id=task.task_id, task_name=task.name,
        reward=round(reward, 2), max_possible_reward=max_reward,
        score_pct=round(score_pct, 1), passed=passed,
        feedback=feedback,
        breakdown={
            "required_hit": list(got_required),
            "bonus_hit": list(got_bonus),
            "penalised_hit": list(got_penalised),
            "missing_required": list(required - selected),
        },
    )


# ---------------------------------------------------------------------------
# GRADER 2 — Chart Selection
# ---------------------------------------------------------------------------

def grade_chart_selection(action: List[str], task: Task) -> GradeResult:
    """
    action: list of chart type strings the agent chose.
    """
    cfg = task.reward_config
    expected = task.expected_output
    best       = set(expected["best"])
    acceptable = set(expected["acceptable"])
    poor       = set(expected["poor"])

    max_reward = (
        len(best) * cfg["per_best"]
        + len(acceptable) * cfg["per_acceptable"]
        + cfg["all_best_bonus"]
    )

    if not isinstance(action, list):
        return GradeResult(
            task_id=task.task_id, task_name=task.name,
            reward=-1.0, max_possible_reward=max_reward,
            score_pct=0.0, passed=False,
            feedback="Action must be a list of chart type strings.",
        )

    selected = set(action)
    got_best       = selected & best
    got_acceptable = selected & acceptable
    got_poor       = selected & poor

    reward = 0.0
    reward += len(got_best)       * cfg["per_best"]
    reward += len(got_acceptable) * cfg["per_acceptable"]
    reward += len(got_poor)       * cfg["per_poor"]

    if best.issubset(selected) and not got_poor:
        reward += cfg["all_best_bonus"]
        feedback = "Excellent! All best charts chosen with no poor choices."
        passed = True
    elif got_best:
        feedback = f"Chose {len(got_best)}/{len(best)} best charts."
        passed = True
    else:
        feedback = "No optimal charts selected. Reconsider chart types."
        passed = False

    score_pct = max(0.0, min(100.0, (reward / max_reward) * 100))

    return GradeResult(
        task_id=task.task_id, task_name=task.name,
        reward=round(reward, 2), max_possible_reward=max_reward,
        score_pct=round(score_pct, 1), passed=passed,
        feedback=feedback,
        breakdown={
            "best_hit": list(got_best),
            "acceptable_hit": list(got_acceptable),
            "poor_hit": list(got_poor),
        },
    )


# ---------------------------------------------------------------------------
# GRADER 3 — Insight Generation
# ---------------------------------------------------------------------------

def grade_insight_generation(action: List[str], task: Task) -> GradeResult:
    """
    action: list of insight strings generated by the agent.
    Scores based on keyword coverage across themes.
    """
    cfg = task.reward_config
    expected = task.expected_output
    keyword_map: Dict[str, List[str]] = expected["keyword_map"]
    required_themes: List[str] = expected["required_themes"]
    min_ins = expected["min_insights"]
    max_ins = expected["max_insights"]

    max_reward = cfg["max_reward"]

    if not isinstance(action, list) or not all(isinstance(i, str) for i in action):
        return GradeResult(
            task_id=task.task_id, task_name=task.name,
            reward=-1.0, max_possible_reward=max_reward,
            score_pct=0.0, passed=False,
            feedback="Action must be a list of insight strings.",
        )

    combined_text = " ".join(action).lower()
    covered, missing = [], []

    reward = 0.0
    for theme in required_themes:
        keywords = keyword_map[theme]
        if any(kw in combined_text for kw in keywords):
            covered.append(theme)
            reward += cfg["per_theme_covered"]
        else:
            missing.append(theme)
            reward += cfg["length_penalty_per_missing"]

    # Count bonus/penalty for number of insights
    n = len(action)
    if min_ins <= n <= max_ins:
        reward += cfg["per_insight_in_range"]
        count_note = f"Insight count {n} is in range [{min_ins}–{max_ins}]. ✓"
    else:
        count_note = f"Insight count {n} out of range [{min_ins}–{max_ins}]."

    reward = min(reward, max_reward)
    score_pct = max(0.0, (reward / max_reward) * 100)
    passed = len(covered) >= len(required_themes) * 0.6

    feedback = (
        f"Themes covered: {covered}. Missing: {missing}. {count_note}"
    )

    return GradeResult(
        task_id=task.task_id, task_name=task.name,
        reward=round(reward, 2), max_possible_reward=max_reward,
        score_pct=round(score_pct, 1), passed=passed,
        feedback=feedback,
        breakdown={
            "themes_covered": covered,
            "themes_missing": missing,
            "num_insights": n,
        },
    )


# ---------------------------------------------------------------------------
# Master Grader — dispatches to the right grader by task_id
# ---------------------------------------------------------------------------

GRADER_MAP = {
    0: grade_column_classification,
    1: grade_kpi_selection,
    2: grade_chart_selection,
    3: grade_insight_generation,
}


def grade(action: Any, task: Task) -> GradeResult:
    """
    Grade an agent action against the given task.
    Dispatches to the appropriate grader function.
    """
    grader_fn = GRADER_MAP.get(task.task_id)
    if grader_fn is None:
        raise ValueError(f"No grader registered for task_id={task.task_id}")
    return grader_fn(action, task)


def total_score(results: List[GradeResult]) -> Tuple[float, float, float]:
    """
    Aggregate multiple GradeResults.
    Returns (total_reward, max_reward, overall_pct).
    """
    total  = sum(r.reward for r in results)
    max_r  = sum(r.max_possible_reward for r in results)
    pct    = (total / max_r * 100) if max_r else 0.0
    return round(total, 2), round(max_r, 2), round(pct, 1)


if __name__ == "__main__":
    # Quick smoke-test
    from tasks import get_task

    dummy_actions = [
        {  # Task 0 — perfect classification
            "date": "datetime", "product": "categorical",
            "category": "categorical", "region": "categorical",
            "sales": "numerical", "units_sold": "numerical",
            "profit": "numerical", "customer_age": "numerical",
            "customer_gender": "categorical",
        },
        ["total_sales", "total_profit", "profit_margin", "sales_growth", "top_product"],  # Task 1
        ["line_chart", "bar_chart"],          # Task 2
        [                                     # Task 3
            "Sales have increased steadily over the months January to March.",
            "Electronics is the top performing category with highest revenue.",
            "The North region leads in total sales and profit.",
            "Profit margins are higher for Electronics compared to Furniture.",
            "Laptop is the best-selling product across all regions.",
        ],
    ]

    results = []
    for task_id, action in enumerate(dummy_actions):
        task = get_task(task_id)
        result = grade(action, task)
        print(result)
        results.append(result)

    total, max_r, pct = total_score(results)
    print(f"\n{'='*55}")
    print(f"OVERALL  →  {total} / {max_r}  ({pct}%)")
    print(f"{'='*55}")