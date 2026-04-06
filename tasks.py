"""
tasks.py — Defines all RL tasks, valid actions, and expected outputs
for the Data Analyst RL Environment.

Each task represents one step in the data analysis pipeline:
  Step 0: Column Classification
  Step 1: KPI Selection
  Step 2: Chart Selection
  Step 3: Insight Generation
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


# ---------------------------------------------------------------------------
# Task Definitions
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """Encapsulates a single RL task."""
    task_id: int
    name: str
    description: str
    valid_actions: List[Any]
    expected_output: Any
    reward_config: Dict[str, float]
    hint: str = ""


# ------------------------------------------------------------------
# TASK 0 — Column Classification
# ------------------------------------------------------------------
COLUMN_CLASSIFICATION_TASK = Task(
    task_id=0,
    name="Column Classification",
    description=(
        "Classify each column of the dataset into one of three types: "
        "'numerical', 'categorical', or 'datetime'."
    ),
    valid_actions=[
        # Each action is a dict mapping column_name -> type
        {
            "date": "datetime",
            "product": "categorical",
            "category": "categorical",
            "region": "categorical",
            "sales": "numerical",
            "units_sold": "numerical",
            "profit": "numerical",
            "customer_age": "numerical",
            "customer_gender": "categorical",
        }
    ],
    expected_output={
        "date": "datetime",
        "product": "categorical",
        "category": "categorical",
        "region": "categorical",
        "sales": "numerical",
        "units_sold": "numerical",
        "profit": "numerical",
        "customer_age": "numerical",
        "customer_gender": "categorical",
    },
    reward_config={
        "perfect_match": 5.0,
        "per_correct": 0.5,
        "per_wrong": -0.5,
        "partial_threshold": 0.6,   # fraction correct for partial reward
        "partial_reward": 1.5,
    },
    hint="Look at dtype: int/float → numerical, object/bool → categorical, datetime → datetime.",
)


# ------------------------------------------------------------------
# TASK 1 — KPI Selection
# ------------------------------------------------------------------
KPI_SELECTION_TASK = Task(
    task_id=1,
    name="KPI Selection",
    description=(
        "Select the most relevant KPIs to compute for this dataset. "
        "Choose from: ['total_sales', 'average_sales', 'total_profit', "
        "'average_profit', 'profit_margin', 'sales_growth', 'top_product', "
        "'top_region', 'units_per_transaction']"
    ),
    valid_actions=[
        ["total_sales", "total_profit", "profit_margin", "sales_growth", "top_product"],
        ["total_sales", "average_sales", "total_profit", "profit_margin", "top_region"],
        ["total_sales", "total_profit", "sales_growth", "top_product", "top_region"],
    ],
    expected_output={
        "required": ["total_sales", "total_profit", "profit_margin"],
        "bonus":    ["sales_growth", "top_product", "top_region"],
        "penalised": ["units_per_transaction"],   # irrelevant for this dataset
    },
    reward_config={
        "per_required": 1.5,
        "per_bonus": 0.5,
        "per_penalised": -1.0,
        "perfect_reward": 3.0,
    },
    hint="Focus on revenue, profitability, and trend KPIs for a sales dataset.",
)


# ------------------------------------------------------------------
# TASK 2 — Chart Selection
# ------------------------------------------------------------------

CHART_OPTIONS = [
    "bar_chart",
    "line_chart",
    "histogram",
    "scatter_plot",
    "pie_chart",
    "heatmap",
    "box_plot",
    "area_chart",
]

CHART_SELECTION_TASK = Task(
    task_id=2,
    name="Chart Selection",
    description=(
        "Choose the best chart type(s) to visualise the dataset insights. "
        f"Available options: {CHART_OPTIONS}"
    ),
    valid_actions=[
        ["line_chart", "bar_chart"],
        ["bar_chart", "line_chart", "pie_chart"],
        ["line_chart", "bar_chart", "histogram"],
    ],
    expected_output={
        "best": ["line_chart", "bar_chart"],      # temporal trend + category comparison
        "acceptable": ["area_chart", "pie_chart"],
        "poor": ["scatter_plot", "heatmap", "box_plot"],
    },
    reward_config={
        "per_best": 2.0,
        "per_acceptable": 0.5,
        "per_poor": -1.0,
        "all_best_bonus": 1.0,
    },
    hint=(
        "Sales over time → line chart; "
        "category/region comparison → bar chart; "
        "distribution → histogram."
    ),
)


# ------------------------------------------------------------------
# TASK 3 — Insight Generation
# ------------------------------------------------------------------

INSIGHT_KEYWORDS = {
    "trend":        ["increas", "decreas", "grew", "declined", "trend", "over time"],
    "top_performer":["highest", "top", "best", "leading", "most"],
    "profitability":["profit", "margin", "revenue", "earning"],
    "category":     ["electronic", "furniture", "category", "segment"],
    "region":       ["north", "south", "east", "west", "region"],
}

INSIGHT_GENERATION_TASK = Task(
    task_id=3,
    name="Insight Generation",
    description=(
        "Generate 3–5 human-readable business insights from the dataset. "
        "Each insight should be a concise sentence describing a pattern, "
        "trend, or anomaly found in the data."
    ),
    valid_actions=[],   # free-text; grader scores via keyword matching
    expected_output={
        "required_themes": list(INSIGHT_KEYWORDS.keys()),
        "min_insights": 3,
        "max_insights": 5,
        "keyword_map": INSIGHT_KEYWORDS,
    },
    reward_config={
        "per_theme_covered": 1.0,
        "per_insight_in_range": 0.5,
        "length_penalty_per_missing": -0.5,
        "max_reward": 6.0,
    },
    hint=(
        "Cover: sales trend over months, top product, top region, "
        "profit margins by category, and Electronics vs Furniture comparison."
    ),
)


# ------------------------------------------------------------------
# Task Registry
# ------------------------------------------------------------------

ALL_TASKS: List[Task] = [
    COLUMN_CLASSIFICATION_TASK,
    KPI_SELECTION_TASK,
    CHART_SELECTION_TASK,
    INSIGHT_GENERATION_TASK,
]

TASK_NAMES = {t.task_id: t.name for t in ALL_TASKS}


def get_task(task_id: int) -> Optional[Task]:
    """Return a Task by its ID, or None if not found."""
    for task in ALL_TASKS:
        if task.task_id == task_id:
            return task
    return None


def describe_all_tasks() -> str:
    """Return a human-readable summary of all tasks."""
    lines = ["=" * 60, "DATA ANALYST RL ENVIRONMENT — TASK REGISTRY", "=" * 60]
    for task in ALL_TASKS:
        lines.append(f"\n[Task {task.task_id}] {task.name}")
        lines.append(f"  Description : {task.description}")
        lines.append(f"  Hint        : {task.hint}")
        lines.append(f"  Rewards     : {task.reward_config}")
    lines.append("=" * 60)
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe_all_tasks())