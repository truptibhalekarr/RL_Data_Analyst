<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:6a85b6,50:bac8e0,100:dde1f4&height=180&section=header&text=RL%20Data%20Analyst&fontSize=60&fontColor=ffffff&fontAlignY=42&desc=Reinforcement%20Learning%20for%20Automated%20Data%20Analysis&descAlignY=62&descSize=16&animation=fadeIn" />

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge\&logo=python\&logoColor=white)](https://python.org)
[![RL](https://img.shields.io/badge/Reinforcement-Learning-blueviolet?style=for-the-badge)]()
[![Status](https://img.shields.io/badge/Status-Active-6a85b6?style=for-the-badge)]()

<br/>

> *An AI agent that learns to perform end-to-end data analysis — step by step.*

</div>

---

## 🌐 Live Demo

👉 https://huggingface.co/spaces/truptibhalekarr/RL_Data_Analyst_Agent

---

## 🔮 What is RL Data Analyst?

**RL Data Analyst** is a custom-built Reinforcement Learning environment where an AI agent learns how to perform **real-world data analysis tasks**.

Instead of static pipelines, the agent:

* Makes decisions step-by-step
* Learns from rewards & penalties
* Optimizes for analytical quality

👉 It mimics how a human data analyst thinks.

---

## ✨ Features

| Feature                          | Description                                                        |
| -------------------------------- | ------------------------------------------------------------------ |
| 🤖 **RL-based Pipeline**         | Entire data analysis flow modeled as a sequential decision problem |
| 📊 **Auto Column Understanding** | Detects numerical, categorical, datetime columns                   |
| 📈 **Smart KPI Selection**       | Identifies business metrics like revenue, growth                   |
| 📉 **Chart Intelligence**        | Selects best chart type based on data                              |
| 💡 **Insight Generation**        | Produces human-readable business insights                          |
| 🧾 **Reward System**             | Scores each step based on correctness                              |

---

## 🧩 The 4-Step RL Workflow

```
Step 0 → Column Classification
Step 1 → KPI Selection
Step 2 → Chart Selection
Step 3 → Insight Generation
```

---

## 🧠 How RL Works Here

```
STATE   → Dataset schema (columns, types, distributions)
ACTION  → Agent decision at each step
REWARD  → Based on correctness of decision
GOAL    → Maximize total analysis quality
```

---

## 🏆 Benchmark Results

| Agent             | Score    | Result |
| ----------------- | -------- | ------ |
| 🥇 HeuristicAgent | 95%+     | ✅ PASS |
| 🥈 RuleBasedAgent | 95%+     | ✅ PASS |
| 🥉 RandomAgent    | Negative | ❌ FAIL |

---

## 🧠 Agent Types

### 🎲 RandomAgent

Completely random decisions — baseline performance.

### 📏 RuleBasedAgent

Uses dataset patterns and heuristics.

### 🧠 HeuristicAgent

Advanced logic + confidence scoring → best performance.

---

## 🧪 Sample Output

```
[START] task=Column Classification
[STEP] step=1 reward=9.5
[STEP] step=2 reward=9.0
[STEP] step=3 reward=5.0
[STEP] step=4 reward=5.5
[END] task=Insight Generation score=91.7 steps=4
```

---

## 📁 Project Structure

```
RL_Data_Analyst/
│
├── core/            # Analysis engines
├── data/            # Dataset
├── env.py           # RL environment
├── agent.py         # Agents
├── grader.py        # Reward system
├── inference.py     # API + evaluator entrypoint
├── tasks.py         # Task definitions
├── dataset.csv      # Sample dataset
└── requirements.txt
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/truptibhalekarr/RL_Data_Analyst
cd RL_Data_Analyst

pip install -r requirements.txt

python main.py
```

---

## 🛠️ Tech Stack

```python
stack = {
    "Language" : "Python 3.10+",
    "Data"     : "Pandas + NumPy",
    "RL Env"   : "Custom-built (no external library)",
    "API"      : "FastAPI",
    "Deploy"   : "Hugging Face Spaces"
}
```

---

## 🧠 Why This Matters

Traditional pipelines:
❌ Static
❌ Rule-based
❌ Non-adaptive

RL approach:
✅ Dynamic
✅ Learns from feedback
✅ Generalizes better

👉 This is closer to **next-gen AI data systems**

---

## 📌 Hackathon Context

Built for a hackathon requiring:

* RL environment design
* Reward logic
* Task-based evaluation
* Real-world use case

---

## 👩‍💻 Author

**Trupti Bhalekar**

* GitHub: https://github.com/truptibhalekarr
* Hugging Face: https://huggingface.co/truptibhalekarr

---

<div align="center">

⭐ *Star this repo if you liked the project!*

</div>
