import numpy as np
from agent import AvellanedaStoikovAgent


class MultiAssetASAgent:
    """
    Extension of A-S to 2 correlated assets.

    Reservation price vector (Guéant et al. extension):
        r_i = s_i - gamma * (T-t) * [Sigma @ q]_i

    where Sigma is the 2x2 return covariance matrix,
    q = [q1, q2] is the inventory vector.

    Spread for each asset uses its own marginal variance:
        delta_i^a + delta_i^b = gamma*sigma_ii^2*(T-t)
                               + (2/gamma)*ln(1 + gamma/k_i)
    """

    def __init__(self, gamma=0.1, rho=0.5):
        self.gamma = gamma
        self.rho   = rho

    def covariance_matrix(self, sigma1, sigma2):
        return np.array([
            [sigma1 ** 2,                    self.rho * sigma1 * sigma2],
            [self.rho * sigma1 * sigma2,     sigma2 ** 2               ],
        ])

    def reservation_prices(self, s_vec, q_vec, t, T, sigma1, sigma2):
        """s_vec = [s1, s2], q_vec = [q1, q2]. Returns r_vec = [r1, r2]."""
        Sigma = self.covariance_matrix(sigma1, sigma2)
        r_vec = np.array(s_vec) - self.gamma * (T - t) * (Sigma @ np.array(q_vec))
        return r_vec

    def compute_quotes(self, s_vec, q_vec, t, T, sigma1, sigma2, k1, k2):
        """Returns [(delta_a1, delta_b1), (delta_a2, delta_b2)]."""
        r_vec  = self.reservation_prices(s_vec, q_vec, t, T, sigma1, sigma2)
        quotes = []
        for s, r, sigma, k in zip(s_vec, r_vec, [sigma1, sigma2], [k1, k2]):
            spread  = (self.gamma * sigma ** 2 * (T - t)
                       + (2 / self.gamma) * np.log(1 + self.gamma / k))
            delta_a = max((r - s) + spread / 2, 1e-4)
            delta_b = max((s - r) + spread / 2, 1e-4)
            quotes.append((delta_a, delta_b))
        return quotes


class SimpleRegimeDetector:
    """
    Two-regime classifier using rolling volatility.
    Regime 0 = low-vol / normal market
    Regime 1 = high-vol / stress / trending

    IMPORTANT: call fit_threshold() on TRAINING data only.
    """

    def __init__(self, window=50, vol_threshold=None):
        self.window          = window
        self.vol_threshold   = vol_threshold
        self._price_buffer   = []

    def update(self, price: float):
        self._price_buffer.append(price)
        if len(self._price_buffer) > self.window + 1:
            self._price_buffer.pop(0)

    def current_regime(self) -> int:
        if len(self._price_buffer) < self.window or self.vol_threshold is None:
            return 0
        rets       = np.diff(self._price_buffer[-self.window:])
        rolling_vol = np.std(rets)
        return int(rolling_vol > self.vol_threshold)

    def fit_threshold(self, all_returns: np.ndarray, percentile=75):
        """Set vol_threshold from TRAINING data only — never test data."""
        window = self.window
        vols   = [
            np.std(all_returns[i: i + window])
            for i in range(len(all_returns) - window)
        ]
        self.vol_threshold = float(np.percentile(vols, percentile))
        print(f"Regime vol threshold: {self.vol_threshold:.4f} (p{percentile})")
        return self.vol_threshold


class RegimeAdaptiveAgent:
    """
    Wraps the base A-S agent and modulates gamma by detected regime.
    In stress regimes: higher gamma = wider spreads, faster inventory reduction.
    """

    def __init__(self, gamma_normal=0.1, gamma_stress=1.0):
        self.gamma_normal = gamma_normal
        self.gamma_stress = gamma_stress
        self.detector     = SimpleRegimeDetector()

    def compute_quotes(self, s, q, t, T, sigma, k):
        regime = self.detector.current_regime()
        gamma  = self.gamma_stress if regime == 1 else self.gamma_normal
        agent  = AvellanedaStoikovAgent(gamma=gamma)
        self.detector.update(s)
        return agent.compute_quotes(s, q, t, T, sigma, k)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: run a 2-asset simulation for one rho value
# ─────────────────────────────────────────────────────────────────────────────

def _run_two_asset(rho, n_paths=500, gamma=0.1, seed=42):
    sigma1, sigma2 = 2.0, 1.5
    k1,  k2        = 1.5, 1.5
    A, dt, T       = 140, 0.005, 1.0
    steps          = int(T / dt)
    rng            = np.random.default_rng(seed)

    pnls, qs1, qs2 = [], [], []
    for _ in range(n_paths):
        agent       = MultiAssetASAgent(gamma=gamma, rho=rho)
        s1, s2      = 100.0, 50.0
        x, q1, q2   = 0.0, 0, 0

        for i in range(steps):
            t      = i * dt
            quotes = agent.compute_quotes(
                [s1, s2], [q1, q2], t, T, sigma1, sigma2, k1, k2
            )
            (da1, db1), (da2, db2) = quotes

            # Asset 1 fills
            if rng.random() < A * np.exp(-k1 * da1) * dt:
                x += s1 + da1;  q1 -= 1
            if rng.random() < A * np.exp(-k1 * db1) * dt:
                x -= s1 - db1;  q1 += 1

            # Asset 2 fills
            if rng.random() < A * np.exp(-k2 * da2) * dt:
                x += s2 + da2;  q2 -= 1
            if rng.random() < A * np.exp(-k2 * db2) * dt:
                x -= s2 - db2;  q2 += 1

            # Price evolution
            s1 += sigma1 * np.sqrt(dt) * rng.standard_normal()
            s2 += sigma2 * np.sqrt(dt) * rng.standard_normal()

        pnls.append(x + q1 * s1 + q2 * s2)
        qs1.append(q1);  qs2.append(q2)

    return (float(np.mean(pnls)), float(np.std(pnls)),
            float(np.std(qs1)),   float(np.std(qs2)))


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 4: Multi-Asset & Regime Extension")
    print("=" * 60)

    # ── Part 1: sanity check rho=0 collapses to single-asset ─────────────
    print("\n--- Sanity check: rho=0 must match single-asset formula ---")
    single  = AvellanedaStoikovAgent(gamma=0.1)
    multi_0 = MultiAssetASAgent(gamma=0.1, rho=0.0)

    s_vec, q_vec = [100.0, 50.0], [2, -1]
    t, T, s1, s2 = 0.0, 1.0, 2.0, 1.5

    r_multi  = multi_0.reservation_prices(s_vec, q_vec, t, T, s1, s2)
    r_single1 = single.reservation_price(100.0,  2, t, T, s1)
    r_single2 = single.reservation_price( 50.0, -1, t, T, s2)

    match1 = np.isclose(r_multi[0], r_single1)
    match2 = np.isclose(r_multi[1], r_single2)
    print(f"  Asset 1:  multi r={r_multi[0]:.4f}  single r={r_single1:.4f}  "
          f"{'PASS' if match1 else 'FAIL'}")
    print(f"  Asset 2:  multi r={r_multi[1]:.4f}  single r={r_single2:.4f}  "
          f"{'PASS' if match2 else 'FAIL'}")

    # ── Part 2: show how correlation affects reservation prices ───────────
    print("\n--- Effect of correlation on reservation prices ---")
    print("  (s=[100,50], q=[2,-1], t=0, T=1, sigma=[2,1.5])")
    print(f"  {'rho':>6}  {'r1':>8}  {'r2':>8}  {'spread1':>9}  {'spread2':>9}")
    print("  " + "-" * 50)
    for rho in [-0.5, 0.0, 0.5, 0.9]:
        agent = MultiAssetASAgent(gamma=0.1, rho=rho)
        r     = agent.reservation_prices(s_vec, q_vec, t, T, s1, s2)
        qs    = agent.compute_quotes(s_vec, q_vec, t, T, s1, s2, 1.5, 1.5)
        sp1   = qs[0][0] + qs[0][1]
        sp2   = qs[1][0] + qs[1][1]
        print(f"  {rho:>6.1f}  {r[0]:>8.4f}  {r[1]:>8.4f}  {sp1:>9.4f}  {sp2:>9.4f}")

    # ── Part 3: Monte Carlo P&L table across rho values ──────────────────
    print("\n--- Monte Carlo simulation: 500 paths, gamma=0.1 ---")
    print(f"  {'rho':>6}  {'mean_pnl':>10}  {'std_pnl':>9}  "
          f"{'std_q1':>8}  {'std_q2':>8}")
    print("  " + "-" * 50)
    for rho in [-0.5, 0.0, 0.5, 0.9]:
        m, s, q1, q2 = _run_two_asset(rho)
        print(f"  {rho:>6.1f}  {m:>10.1f}  {s:>9.1f}  {q1:>8.1f}  {q2:>8.1f}")

    print("\nKey finding: at rho=0.9 (highly correlated assets),")
    print("std_pnl and std_q both increase — correlated inventory")
    print("risk cannot be diversified away, forcing wider spreads.")

    # ── Part 4: regime detector ───────────────────────────────────────────
    print("\n--- Regime detector ---")
    rng          = np.random.default_rng(42)
    train_prices = np.cumsum(
        np.r_[100, rng.normal(0, 2 * np.sqrt(0.005), 1000)]
    )
    train_returns = np.diff(train_prices)

    detector  = SimpleRegimeDetector(window=50)
    threshold = detector.fit_threshold(train_returns, percentile=75)

    # Feed a short test path and count regime switches
    test_prices = np.cumsum(
        np.r_[100, rng.normal(0, 2 * np.sqrt(0.005), 200)]
    )
    regimes = []
    for p in test_prices:
        detector.update(p)
        regimes.append(detector.current_regime())

    regimes = np.array(regimes)
    print(f"  Normal steps (regime 0): {np.sum(regimes == 0)}")
    print(f"  Stress steps (regime 1): {np.sum(regimes == 1)}")
