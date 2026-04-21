import numpy as np
from market_sim import MarketSimulator
from agent import AvellanedaStoikovAgent, SymmetricAgent


def run_experiment(gamma, n_paths=1000, seed=42):
    sim = MarketSimulator()
    inv = AvellanedaStoikovAgent(gamma=gamma)
    rng = np.random.default_rng(seed)

    inv_results = []
    for _ in range(n_paths):
        res = sim.simulate_path(inv, rng)
        inv_results.append(res)

    # FIX (was bug 1): time-average the spread over all timesteps [0, T].
    # The paper's symmetric agent uses the inventory strategy's average spread
    # over the whole simulation — NOT just the value at t=0.
    # At t=0 the spread is at its maximum (gamma*sigma^2*T + const).
    # It shrinks toward (const) as t->T, so the t=0 value overstates the
    # average, especially at high gamma. Using np.linspace + np.mean gives
    # the correct time-average that matches the paper's "Average spread" column.
    times = np.linspace(0, sim.T, sim.steps)
    avg_spread = np.mean([
        inv.optimal_spread(t, sim.T, sim.sigma, sim.k) for t in times
    ])

    sym = SymmetricAgent(half_spread=avg_spread / 2)
    rng2 = np.random.default_rng(seed)
    sym_results = []
    for _ in range(n_paths):
        res = sim.simulate_path(sym, rng2)
        sym_results.append(res)

    def stats(results):
        pnls   = np.array([r["pnl"]     for r in results])
        inv_qs = np.array([r["final_q"] for r in results])
        return {
            "mean_pnl": np.mean(pnls),
            "std_pnl":  np.std(pnls),
            "mean_q":   np.mean(inv_qs),
            "std_q":    np.std(inv_qs),
        }

    # FIX (was bug 3): return avg_spread so main() reuses the same value
    # instead of recomputing it separately (which was also using t=0).
    return stats(inv_results), stats(sym_results), avg_spread


def main():
    print("=" * 60)
    print("Replicating Avellaneda & Stoikov (2008) Tables 1–3")
    print("=" * 60)
    print(f"{'gamma':>6}  {'Strategy':>10}  {'Avg spread':>10}  "
          f"{'Profit':>8}  {'Std':>6}  {'Final q':>8}  {'Std q':>6}")
    print("-" * 70)

    for gamma in [0.01, 0.1, 1.0]:
        # FIX (was bug 2): use avg_spread returned from run_experiment,
        # not a separately recomputed t=0 value.
        inv_s, sym_s, avg_spread = run_experiment(gamma)

        print(f"{gamma:>6.2f}  {'Inventory':>10}  {avg_spread:>10.2f}  "
              f"{inv_s['mean_pnl']:>8.1f}  {inv_s['std_pnl']:>6.1f}  "
              f"{inv_s['mean_q']:>8.2f}  {inv_s['std_q']:>6.1f}")
        print(f"{'':>6}  {'Symmetric':>10}  {avg_spread:>10.2f}  "
              f"{sym_s['mean_pnl']:>8.1f}  {sym_s['std_pnl']:>6.1f}  "
              f"{sym_s['mean_q']:>8.2f}  {sym_s['std_q']:>6.1f}")
        print()

    print("Expected (from paper):")
    print("  gamma=0.1:  Inventory pnl~65 std~6.6  | Symmetric pnl~68 std~12.7")
    print("  gamma=0.01: both strategies similar   | gamma=1.0: wider spread, lower pnl")


if __name__ == "__main__":
    main()
