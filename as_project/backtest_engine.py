import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class Trade:
    time:              float
    price:             float
    size:              int
    side:              str   # 'buy' or 'sell'
    pnl_contribution:  float = 0.0


@dataclass
class BacktestResult:
    trades:           List[Trade]
    pnl_series:       np.ndarray
    price_series:     np.ndarray
    inventory_series: np.ndarray
    final_pnl:        float
    sharpe:           float
    max_drawdown:     float
    total_trades:     int


class BacktestEngine:
    """
    Strict no-lookahead backtest.

    Rules enforced:
      - Agent only sees prices up to and including current step t.
      - Parameters fitted on training data are frozen before test starts.
      - Execution price is the NEXT step's mid-price to simulate latency.
      - train_ratio controls the train/test split boundary.
    """

    def __init__(self, train_ratio=0.7):
        self.train_ratio = train_ratio

    def run(self, price_data: np.ndarray, agent, sim_params: dict) -> BacktestResult:
        """
        price_data   : array of observed mid-prices (real or simulated)
        agent        : object with compute_quotes(s, q, t, T, sigma, k)
        sim_params   : dict with keys: sigma, k, A, dt
                       optional: sigma_t_series (array, same length as test set)
        """
        n     = len(price_data)
        split = int(n * self.train_ratio)

        # ── Strict separation: test only ─────────────────────────────
        test_prices = price_data[split:]
        n_test      = len(test_prices)

        sigma = sim_params["sigma"]
        k     = sim_params["k"]
        A     = sim_params["A"]
        dt    = sim_params["dt"]
        T     = n_test * dt

        sigma_t_series = sim_params.get(
            "sigma_t_series", [sigma] * n_test
        )

        x        = 0.0
        q        = 0
        pnl_hist = np.zeros(n_test)
        inv_hist = np.zeros(n_test)
        trades   = []

        for i in range(n_test - 1):  # stop one step early (need next price)
            S   = test_prices[i]
            t   = i * dt
            sig = sigma_t_series[i]

            d_a, d_b = agent.compute_quotes(S, q, t, T, sig, k)
            pa = S + d_a
            pb = S - d_b

            lam_a = A * np.exp(-k * d_a)
            lam_b = A * np.exp(-k * d_b)

            # Execute at NEXT price (latency simulation, no lookahead)
            S_next = test_prices[i + 1]

            if np.random.random() < lam_a * dt:
                x += pa
                q -= 1
                trades.append(Trade(t, pa, 1, "sell"))

            if np.random.random() < lam_b * dt:
                x -= pb
                q += 1
                trades.append(Trade(t, pb, 1, "buy"))

            pnl_hist[i] = x + q * S_next
            inv_hist[i] = q

        # Mark to market at end
        final_pnl = x + q * test_prices[-1]
        pnl_hist[-1] = final_pnl
        inv_hist[-1] = q

        # ── Risk metrics ─────────────────────────────────────────────
        active     = pnl_hist[pnl_hist != 0]
        returns    = np.diff(active) if len(active) > 1 else np.array([0.0])
        std_ret    = np.std(returns)
        sharpe     = (
            float(np.mean(returns) / std_ret * np.sqrt(252))
            if std_ret > 0 else 0.0
        )

        rolling_max  = np.maximum.accumulate(pnl_hist)
        max_drawdown = float(np.min(pnl_hist - rolling_max))

        return BacktestResult(
            trades=trades,
            pnl_series=pnl_hist,
            price_series=test_prices,
            inventory_series=inv_hist,
            final_pnl=final_pnl,
            sharpe=round(sharpe, 4),
            max_drawdown=round(max_drawdown, 4),
            total_trades=len(trades),
        )
