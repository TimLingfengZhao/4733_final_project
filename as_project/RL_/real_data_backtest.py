"""
Real-data backtest: RL Agent vs A-S Inventory vs Symmetric.

Price dynamics come from real AAPL 2012-06-21 LOBSTER mid-prices.
Prices are normalised to S0=100 so that the paper-scale parameters
(sigma~2, k=1.5, A=140) apply consistently across all three agents.
This ensures the RL agent — trained in the same abstract unit space —
is evaluated under identical conditions to A-S and Symmetric.

Outputs
-------
RL_/pnl_comparison.png    — cumulative P&L curves (3 strategies)
RL_/rolling_sharpe.png    — rolling Sharpe ratio (window=500 steps)
Terminal table             — Final P&L / Sharpe / Max DD / Trades / Mean|q|
"""
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from agent import AvellanedaStoikovAgent, SymmetricAgent
from backtest_engine import BacktestEngine
from lobster_loader import LobsterLoader
from RL_.q_agent import QLearningAgent

PAPER_K = 1.5
PAPER_A = 140.0
TARGET_S0 = 100.0


def load_mid_prices(project_dir: Path, target_points: int = 5000) -> np.ndarray:
    """Load AAPL orderbook, extract mid-price, downsample to ~target_points."""
    data_dir = project_dir / "LOBSTER_SampleFile_AAPL_2012-06-21_10"
    loader = LobsterLoader(str(data_dir), levels=10)
    _, orderbook = loader.load_day("AAPL", "2012-06-21")
    mids = loader.compute_mid_price(orderbook).to_numpy(dtype=float)
    step = max(1, len(mids) // target_points)
    return mids[::step]


def normalise_prices(prices: np.ndarray) -> np.ndarray:
    """Scale prices so first value equals TARGET_S0=100."""
    return prices / prices[0] * TARGET_S0


def estimate_sigma(prices_norm: np.ndarray, dt: float, train_ratio: float = 0.7) -> float:
    """Estimate per-unit-time sigma from normalised training-split prices."""
    split = int(len(prices_norm) * train_ratio)
    train = prices_norm[:split]
    rets = np.diff(train)
    if len(rets) < 2:
        return 2.0
    return float(np.std(rets) / np.sqrt(dt))


def run_backtest(
    price_data: np.ndarray,
    agent,
    sim_params: dict,
    seed: int,
):
    np.random.seed(seed)
    if hasattr(agent, "reset_episode"):
        agent.reset_episode()
    engine = BacktestEngine(train_ratio=0.7)
    return engine.run(price_data, agent, sim_params)


def rolling_sharpe(pnl_series: np.ndarray, window: int = 500) -> np.ndarray:
    rets = np.diff(pnl_series, prepend=pnl_series[0])
    out = np.full_like(pnl_series, fill_value=np.nan, dtype=float)
    for i in range(window, len(rets)):
        win = rets[i - window + 1: i + 1]
        std = np.std(win)
        out[i] = (np.mean(win) / std * np.sqrt(252)) if std > 1e-12 else 0.0
    return out


def save_plots(results: dict, out_dir: Path) -> tuple:
    pnl_path = out_dir / "pnl_comparison.png"
    sharpe_path = out_dir / "rolling_sharpe.png"

    fig, ax = plt.subplots(figsize=(11, 5))
    for name, res in results.items():
        ax.plot(res.pnl_series, label=name)
    ax.set_title("Cumulative P&L — Real AAPL 2012-06-21 (prices normalised to S₀=100)")
    ax.set_xlabel("Test step")
    ax.set_ylabel("P&L ($)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(pnl_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    for name, res in results.items():
        rs = rolling_sharpe(res.pnl_series)
        ax.plot(rs, label=name)
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax.set_title("Rolling Sharpe Ratio (window=500 steps)")
    ax.set_xlabel("Test step")
    ax.set_ylabel("Sharpe")
    ax.legend()
    fig.tight_layout()
    fig.savefig(sharpe_path, dpi=150)
    plt.close(fig)

    return pnl_path, sharpe_path


def build_table(results: dict) -> pd.DataFrame:
    rows = []
    for name, res in results.items():
        rows.append({
            "Strategy":      name,
            "Final P&L":     round(float(res.final_pnl), 4),
            "Sharpe":        round(float(res.sharpe), 4),
            "Max DD":        round(float(res.max_drawdown), 4),
            "Total Trades":  int(res.total_trades),
            "Mean |q|":      round(float(np.mean(np.abs(res.inventory_series))), 4),
        })
    return pd.DataFrame(rows)


def main():
    rl_dir = Path(__file__).resolve().parent
    q_path = rl_dir / "q_table.npy"
    if not q_path.exists():
        raise FileNotFoundError(
            f"Trained Q-table not found: {q_path}\n"
            "Run  python as_project/RL_/train_rl.py  first."
        )

    dt = 0.005
    gamma = 0.1
    seed = 123

    # ── Load and normalise real AAPL prices ──────────────────────────
    raw_prices = load_mid_prices(PROJECT_DIR, target_points=5000)
    prices = normalise_prices(raw_prices)
    sigma = estimate_sigma(prices, dt=dt, train_ratio=0.7)
    print(f"Real AAPL mid-price series: {len(prices)} points "
          f"(normalised S0={prices[0]:.2f})")
    print(f"Estimated sigma (train split): {sigma:.4f}  "
          f"[k={PAPER_K}, A={PAPER_A}]\n")

    sim_params = {"sigma": sigma, "k": PAPER_K, "A": PAPER_A, "dt": dt}

    # ── Instantiate agents ───────────────────────────────────────────
    rl_agent = QLearningAgent(dt=dt, epsilon=0.0)
    rl_agent.load(q_path)

    as_agent = AvellanedaStoikovAgent(gamma=gamma)
    avg_spread = as_agent.optimal_spread(0.0, 1.0, sigma, PAPER_K)
    sym_agent = SymmetricAgent(half_spread=avg_spread / 2)

    # ── Run backtests ────────────────────────────────────────────────
    agent_map = {
        "RL Agent":      rl_agent,
        "A-S Inventory": as_agent,
        "Symmetric":     sym_agent,
    }
    results = {}
    for name, agent in agent_map.items():
        results[name] = run_backtest(prices, agent, sim_params, seed=seed)
        print(f"  Backtested: {name}")

    # ── Save plots ───────────────────────────────────────────────────
    pnl_path, sharpe_path = save_plots(results, rl_dir)

    # ── Print comparison table ───────────────────────────────────────
    table = build_table(results)
    print()
    print("=" * 72)
    print("Real AAPL 2012-06-21 Backtest Comparison")
    print("=" * 72)
    print(table.to_string(index=False))
    print("=" * 72)
    print(f"\nP&L plot         → {pnl_path}")
    print(f"Rolling Sharpe   → {sharpe_path}")


if __name__ == "__main__":
    main()
