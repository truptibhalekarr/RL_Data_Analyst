# 🤖 RL Data Analyst Environment

> A Reinforcement Learning environment where an AI agent learns to perform **automated data analysis** — step by step, just like a human analyst.

---

## 🎯 What This Project Does

This project converts the **entire data analysis pipeline** into an RL problem. At each step, the agent takes an action, and the environment gives it a reward or penalty based on how good that action was.

---

## 🧩 The 4 Tasks (Steps)

| Step | Task | What Agent Decides |
|------|------|--------------------|
| 0 | 📊 Column Classification | numerical / categorical / datetime |
| 1 | 📈 KPI Selection | total_sales, profit_margin, growth... |
| 2 | 📉 Chart Selection | bar chart, line chart, histogram... |
| 3 | 💡 Insight Generation | Human-readable business insights |

---

## 🔁 How RL Works Here

**Examples:**
- ✅ Correct KPI selected → `+1.5` reward
- ✅ Best chart chosen → `+2.0` reward  
- ❌ Wrong chart → `-1.0` penalty
- 🔶 Partial insight → small reward

---

## 🏆 Benchmark Results

| Agent | Avg Score | Result |
|-------|-----------|--------|
| 🥇 HeuristicAgent | **95.1%** | ✅ PASS |
| 🥈 RuleBasedAgent | **95.1%** | ✅ PASS |
| 🥉 RandomAgent | -6.6% | ❌ FAIL |

---

## 📁 Project Structure

rl_data_analyst/
│
├── env.py          # 🌍 RL Environment (reset, step, state)
├── agent.py        # 🤖 3 Agents: Random, RuleBased, Heuristic
├── grader.py       # 🧾 Reward & grading system
├── tasks.py        # 📋 Task definitions & expected outputs
├── main.py         # 🚀 Run and benchmark all agents
├── dataset.csv     # 📊 Sample sales dataset (30 rows)
└── requirements.txt

---

## 🚀 Quick Start
```bash
# 1. Clone the repo
git clone https://github.com/truptibhalekarr/rl-data-analyst.git
cd rl-data-analyst

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run all agents (benchmark)
python main.py

# 4. Run specific agent
python main.py --agent heuristic

# 5. Run quietly (summary only)
python main.py --agent rule --quiet
```

---

## 🧠 Agent Descriptions

### 🎲 RandomAgent
Picks actions completely randomly. Serves as the **baseline** — shows what happens with zero intelligence.

### 📏 RuleBasedAgent  
Uses pandas dtype inspection and column name patterns to make **smart, rule-driven decisions**. Scores 95%+.

### 🧠 HeuristicAgent
Extends RuleBasedAgent with **confidence scoring** and forced required-KPI inclusion. Also logs confidence per step.

---

## 🧪 Sample Output

✅ [0] Column Classification   Reward: +9.50  (100.0%)
✅ [1] KPI Selection           Reward: +9.00  (100.0%)
✅ [2] Chart Selection         Reward: +5.00  (83.3%)
✅ [3] Insight Generation      Reward: +5.50  (91.7%)
🏆 TOTAL : +29.00 / 30.50 (95.1%) — PASS ✅

---

## 🛠️ Tech Stack

- **Python 3.8+**
- **Pandas** — dataset handling
- **NumPy** — numerical operations
- **Custom RL Environment** — no external RL library needed!

---

## 🌐 Live Demo

👉 [Try it on Hugging Face Spaces](https://huggingface.co/spaces/truptibhalekarr/rl-data-analyst)

---

## 📌 Hackathon Submission

This project was built for a hackathon focused on building small RL environments with:
- ✅ Clearly defined tasks
- ✅ Reward system
- ✅ Agent evaluation / grading
- ✅ Real-world problem (data analysis automation)

---

## 👩‍💻 Author

Made with ❤️ by **Trupti Bhalekar**