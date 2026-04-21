"""
Train a tabular Q-learning market-making agent on the MarketSimulator.

Arrival parameters: A is optionally calibrated from LOBSTER AAPL data;
k is fixed at 1.5 (paper default) because the RL action grid and the
paper's simulator use abstract price units where k=1.5 is calibrated.
Mixing real-tick k (~60) with the abstract-unit action grid [0.1..1.2]
would starve the agent of fill signal during training.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from market_sim import MarketSimulator
from lobster_loader import ArrivalRateCalibrator, LobsterLoader
from RL_.q_agent import QLearningAgent

PAPER_A = 140.0
PAPER_K = 1.5


def calibrate_A(cache_path: Path, project_dir: Path) -> float:
    """Calibrate only A from LOBSTER AAPL; k is left at paper default."""
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            cached = json.load(f)
        A_cached = float(cached.get("A", PAPER_A))
        print(f"Loaded cached A={A_cached:.3f} from {cache_path.name}")
        return A_cached

    print("Calibrating A from LOBSTER AAPL data …")
    data_dir = project_dir / "LOBSTER_SampleFile_AAPL_2012-06-21_10"
    loader = LobsterLoader(str(data_dir), levels=10)
    messages, orderbook = loader.load_day("AAPL", "2012-06-21")
    execs = loader.extract_executions(messages, orderbook)
    t_total = float(messages["time"].max() - messages["time"].min())

    calibrator = ArrivalRateCalibrator(n_bins=25)
    results = calibrator.calibrate(execs["delta"].values, t_total)
    A_fit = results["exponential"]["A"]
    A = float(A_fit) if A_fit is not None else PAPER_A

    payload = {
        "A": A,
        "k_used_for_training": PAPER_K,
        "k_raw_from_lobster": results["exponential"]["k"],
        "note": (
            "k is fixed at paper default (1.5) for training. "
            "Raw LOBSTER k is recorded for reference only."
        ),
        "source": "LOBSTER_AAPL_2012-06-21",
    }
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved calibration cache → {cache_path.name}")
    return A


def run_episode(
    sim: MarketSimulator,
    agent: QLearningAgent,
    rng: np.random.Generator,
) -> float:
    S = sim.S0
    x = 0.0
    q = 0
    prices = [S]
    ep_reward = 0.0

    for i in range(sim.steps):
        t = i * sim.dt
        state = agent.build_state(q, t, sim.T, sim.sigma, prices)
        action_idx, (d_a, d_b) = agent.select_action(state, training=True, rng=rng)

        pa = S + d_a
        pb = S - d_b
        lam_a = sim.arrival_intensity(d_a)
        lam_b = sim.arrival_intensity(d_b)

        x_next = x
        q_next = q
        if rng.random() < lam_a * sim.dt:
            x_next += pa
            q_next -= 1
        if rng.random() < lam_b * sim.dt:
            x_next -= pb
            q_next += 1

        S_next = S + sim.sigma * np.sqrt(sim.dt) * rng.standard_normal()
        prices.append(S_next)

        wealth_prev = x + q * S
        wealth_next = x_next + q_next * S_next
        reward = (
            wealth_next
            - wealth_prev
            - agent.inventory_penalty * (q_next ** 2) * sim.dt
        )
        ep_reward += reward

        next_t = min((i + 1) * sim.dt, sim.T)
        next_state = agent.build_state(q_next, next_t, sim.T, sim.sigma, prices)
        agent.update(state, action_idx, reward, next_state)

        x, q, S = x_next, q_next, S_next

    agent.end_episode()
    return float(ep_reward)


def main():
    parser = argparse.ArgumentParser(
        description="Train tabular Q-learning benchmark agent (paper-scale units)."
    )
    parser.add_argument("--episodes", type=int, default=5000,
                        help="Number of training episodes (default: 5000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sigma", type=float, default=2.0,
                        help="Price volatility in paper units (default: 2.0)")
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--no-calibrate", action="store_true",
                        help="Skip LOBSTER calibration; use A=140 (paper default)")
    args = parser.parse_args()

    rl_dir = Path(__file__).resolve().parent
    cache_path = rl_dir / "calibrated_params.json"
    q_table_path = rl_dir / "q_table.npy"

    if args.no_calibrate:
        A = PAPER_A
        print(f"Using paper defaults: A={A:.1f}, k={PAPER_K:.1f}")
    else:
        A = calibrate_A(cache_path, PROJECT_DIR)

    k = PAPER_K  # always use paper k for training
    print(f"Training params: A={A:.3f}, k={k:.3f}, sigma={args.sigma}, "
          f"dt={args.dt}, T={args.T}")

    sim = MarketSimulator(sigma=args.sigma, A=A, k=k, dt=args.dt, T=args.T)
    agent = QLearningAgent(dt=args.dt)
    rng = np.random.default_rng(args.seed)

    rewards = []
    print("=" * 60)
    print(f"Training RL agent (Q-learning) — {args.episodes} episodes")
    print("=" * 60)
    for ep in range(args.episodes):
        agent.reset_episode()
        ep_reward = run_episode(sim, agent, rng)
        rewards.append(ep_reward)
        if (ep + 1) % max(1, args.episodes // 20) == 0 or ep == 0:
            recent = float(np.mean(rewards[max(0, ep - 99): ep + 1]))
            print(
                f"  Episode {ep + 1:6d}/{args.episodes} | "
                f"epsilon={agent.epsilon:.4f} | "
                f"recent_avg_reward={recent:.4f}"
            )

    agent.save(q_table_path)
    initial = float(np.mean(rewards[: min(100, len(rewards))]))
    final = float(np.mean(rewards[max(0, len(rewards) - 100):]))
    print("-" * 60)
    print(f"Episode avg reward: initial {initial:.4f}  →  final {final:.4f}")
    print(f"Q-table saved → {q_table_path}")


if __name__ == "__main__":
    main()
