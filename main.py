"""
main.py — Run and evaluate the Data Analyst RL Environment.

Usage:
    python main.py                         # run all agents, show results
    python main.py --agent heuristic       # run a specific agent
    python main.py --agent random --runs 5 # multiple episodes
"""

import argparse
import sys
from typing import List

from env import DataAnalystEnv
from agent import make_agent, AGENT_REGISTRY, BaseAgent, HeuristicAgent
from grader import total_score, GradeResult


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_episode(env: DataAnalystEnv, agent: BaseAgent, verbose: bool = True) -> List[GradeResult]:
    """Run one episode and return the list of GradeResults."""
    state = env.reset()
    agent.reset()
    results = []

    if verbose:
        print(f"\n🚀  Starting episode  |  Agent: {agent.name}")
        print(state)

    while not state.done:
        action = agent.select_action(state)

        if verbose:
            print(f"\n▶  Step {state.step} — {state.task.name}")
            _print_action(state.step, action)

        state, reward, done, info = env.step(action)
        agent.update(reward, done)
        results.append(info["grade_result"])

        if verbose:
            print(info["grade_result"])

    if verbose:
        print(env.summary())
        if isinstance(agent, HeuristicAgent):
            agent.print_confidence_log()

    return results


def _print_action(step: int, action) -> None:
    """Pretty-print the agent's action."""
    if step == 0 and isinstance(action, dict):
        print(f"  Action (classifications): {action}")
    elif step == 1 and isinstance(action, list):
        print(f"  Action (KPIs): {action}")
    elif step == 2 and isinstance(action, list):
        print(f"  Action (charts): {action}")
    elif step == 3 and isinstance(action, list):
        print("  Action (insights):")
        for i, ins in enumerate(action, 1):
            print(f"    {i}. {ins}")


# ---------------------------------------------------------------------------
# Benchmark — compare all agents
# ---------------------------------------------------------------------------

def benchmark(runs: int = 1, verbose: bool = False) -> None:
    """Run all agents and display a comparison table."""
    env = DataAnalystEnv("dataset.csv")
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║            DATA ANALYST RL — AGENT BENCHMARK                     ║")
    print("╚" + "═" * 68 + "╝")

    summary_rows = []

    for agent_type in AGENT_REGISTRY:
        agent = make_agent(agent_type)
        all_totals, all_pcts = [], []

        for run in range(runs):
            results = run_episode(env, agent, verbose=verbose)
            total, max_r, pct = total_score(results)
            all_totals.append(total)
            all_pcts.append(pct)

        avg_total = sum(all_totals) / len(all_totals)
        avg_pct   = sum(all_pcts)   / len(all_pcts)
        summary_rows.append((agent_type, avg_total, max_r, avg_pct))

    # Print table
    print(f"\n  {'Agent':<18} {'Avg Reward':>12} {'Max Reward':>12} {'Avg Score%':>12}")
    print("  " + "─" * 56)
    for agent_type, avg_t, max_r, avg_p in sorted(summary_rows, key=lambda r: -r[3]):
        bar = "█" * int(avg_p / 10)
        grade = "✅" if avg_p >= 50 else "❌"
        print(
            f"  {agent_type:<18} {avg_t:>12.2f} {max_r:>12.2f} {avg_p:>11.1f}%  "
            f"{grade} {bar}"
        )
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the Data Analyst RL Environment"
    )
    parser.add_argument(
        "--agent",
        choices=list(AGENT_REGISTRY.keys()) + ["all"],
        default="all",
        help="Which agent to run (default: all)",
    )
    parser.add_argument(
        "--runs", type=int, default=1,
        help="Number of episodes per agent (default: 1)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-step output (benchmark mode)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    env  = DataAnalystEnv("dataset.csv")

    if args.agent == "all":
        benchmark(runs=args.runs, verbose=not args.quiet)
    else:
        agent = make_agent(args.agent)
        for run in range(args.runs):
            if args.runs > 1:
                print(f"\n{'─'*60}")
                print(f"  RUN {run + 1} / {args.runs}")
            results = run_episode(env, agent, verbose=not args.quiet)
            if args.quiet:
                total, max_r, pct = total_score(results)
                print(f"[{agent.name}] Run {run+1}: {total:.2f}/{max_r:.2f} ({pct:.1f}%)")


if __name__ == "__main__":
    main()