import numpy as np
import pandas as pd
from pathlib import Path
from agent import AvellanedaStoikovAgent


# ─────────────────────────────────────────────────────────────────────────────
# Real data loader
# ─────────────────────────────────────────────────────────────────────────────

def load_mid_prices(data_dir: str, ticker: str, date: str,
                    levels: int = 10) -> pd.Series:
    """
    Load mid-price time series from LOBSTER orderbook file.
    mid = (best_ask + best_bid) / 2
    Returns pd.Series indexed by timestamp (seconds since midnight).
    """
    folder   = Path(data_dir)
    lob_file = folder / f"{ticker}_{date}_34200000_57600000_orderbook_{levels}.csv"
    msg_file = folder / f"{ticker}_{date}_34200000_57600000_message_{levels}.csv"

    lob_cols = []
    for i in range(1, levels + 1):
        lob_cols += [f"ask_price_{i}", f"ask_size_{i}",
                     f"bid_price_{i}", f"bid_size_{i}"]
    msg_cols = ["time", "type", "order_id", "size", "price", "direction"]

    lob = pd.read_csv(lob_file, header=None, names=lob_cols)
    msg = pd.read_csv(msg_file, header=None, names=msg_cols)

    lob["ask_price_1"] /= 10000
    lob["bid_price_1"] /= 10000
    mid       = (lob["ask_price_1"] + lob["bid_price_1"]) / 2
    mid.index = msg["time"].values
    mid.name  = ticker
    return mid


def align_mid_prices(mid1: pd.Series, mid2: pd.Series,
                     n_points: int = 500) -> tuple:
    """Align two mid-price series onto the same time grid."""
    t_start = max(mid1.index[0],  mid2.index[0])
    t_end   = min(mid1.index[-1], mid2.index[-1])
    grid    = np.linspace(t_start, t_end, n_points)

    def snap(series, grid):
        times  = series.index.values
        prices = series.values
        out    = np.zeros(len(grid))
        for i, t in enumerate(grid):
            idx    = max(np.searchsorted(times, t, side='right') - 1, 0)
            out[i] = prices[idx]
        return out

    return snap(mid1, grid), snap(mid2, grid)


def find_lobster_folder(ticker: str) -> tuple:
    """
    Auto-detect LOBSTER folder for a given ticker in the current directory.
    Folder format: LOBSTER_SampleFile_TICKER_YYYY-MM-DD_LEVELS
    parts[2]=TICKER, parts[3]=DATE after split('_').
    Returns (folder_path_str, date_str) or (None, None).
    """
    candidates = list(Path(".").glob(f"LOBSTER_SampleFile_{ticker}_*"))
    if not candidates:
        return None, None
    parts = candidates[0].name.split("_")
    return str(candidates[0]), parts[3]


def split_intraday(mid: pd.Series) -> tuple:
    """
    Split one stock's mid-price series into morning and afternoon sessions.
    Morning:   9:30am–12:00pm  (34200–43200 seconds since midnight)
    Afternoon: 12:00pm–4:00pm  (43200–57600 seconds)
    Returns (mid_morning, mid_afternoon) as pd.Series.
    """
    NOON = 43200  # 12:00pm in seconds since midnight
    morning   = mid[mid.index <  NOON]
    afternoon = mid[mid.index >= NOON]
    return morning, afternoon


# ─────────────────────────────────────────────────────────────────────────────
# Multi-asset A-S agent
# ─────────────────────────────────────────────────────────────────────────────

class MultiAssetASAgent:
    """
    Extension of A-S to 2 correlated assets.
    Reservation price: r_i = s_i - gamma*(T-t)*[Sigma @ q]_i
    Sigma is the 2x2 return covariance matrix.
    """

    def __init__(self, gamma=0.1, rho=0.5):
        self.gamma = gamma
        self.rho   = rho

    def covariance_matrix(self, sigma1, sigma2):
        return np.array([
            [sigma1**2,               self.rho*sigma1*sigma2],
            [self.rho*sigma1*sigma2,  sigma2**2             ],
        ])

    def reservation_prices(self, s_vec, q_vec, t, T, sigma1, sigma2):
        Sigma = self.covariance_matrix(sigma1, sigma2)
        return np.array(s_vec) - self.gamma*(T-t)*(Sigma @ np.array(q_vec))

    def compute_quotes(self, s_vec, q_vec, t, T, sigma1, sigma2, k1, k2):
        r_vec  = self.reservation_prices(s_vec, q_vec, t, T, sigma1, sigma2)
        quotes = []
        for s, r, sigma, k in zip(s_vec, r_vec, [sigma1, sigma2], [k1, k2]):
            spread  = (self.gamma*sigma**2*(T-t)
                       + (2/self.gamma)*np.log(1 + self.gamma/k))
            delta_a = max((r-s) + spread/2, 1e-4)
            delta_b = max((s-r) + spread/2, 1e-4)
            quotes.append((delta_a, delta_b))
        return quotes


# ─────────────────────────────────────────────────────────────────────────────
# Regime detector
# ─────────────────────────────────────────────────────────────────────────────

class SimpleRegimeDetector:
    """
    Two-regime classifier using rolling volatility of price differences.
    Regime 0 = low-vol / normal.  Regime 1 = high-vol / stress.
    IMPORTANT: call fit_threshold() on TRAINING data only.
    """

    def __init__(self, window=50, vol_threshold=None):
        self.window        = window
        self.vol_threshold = vol_threshold
        self._buf          = []

    def update(self, price: float):
        self._buf.append(price)
        if len(self._buf) > self.window + 1:
            self._buf.pop(0)

    def current_regime(self) -> int:
        if len(self._buf) < self.window or self.vol_threshold is None:
            return 0
        return int(
            np.std(np.diff(self._buf[-self.window:])) > self.vol_threshold
        )

    def fit_threshold(self, prices: np.ndarray, percentile=75):
        """Fit on price array. Call ONLY on training data."""
        w     = self.window
        diffs = np.diff(prices)
        vols  = [np.std(diffs[i:i+w]) for i in range(len(diffs)-w)]
        self.vol_threshold = float(np.percentile(vols, percentile))
        print(f"    Threshold: {self.vol_threshold:.6f} (p{percentile})")
        return self.vol_threshold


class RegimeAdaptiveAgent:
    def __init__(self, gamma_normal=0.1, gamma_stress=1.0):
        self.gamma_normal = gamma_normal
        self.gamma_stress = gamma_stress
        self.detector     = SimpleRegimeDetector()

    def compute_quotes(self, s, q, t, T, sigma, k):
        regime = self.detector.current_regime()
        gamma  = self.gamma_stress if regime == 1 else self.gamma_normal
        self.detector.update(s)
        return AvellanedaStoikovAgent(gamma=gamma).compute_quotes(
            s, q, t, T, sigma, k)


# ─────────────────────────────────────────────────────────────────────────────
# Simulation
# ─────────────────────────────────────────────────────────────────────────────

def simulate_one_path(prices1, prices2, rho, sigma1, sigma2,
                      gamma=0.1, k=1.5, A=140, rng=None):
    """Run multi-asset agent on given price paths. Returns (pnl, q1, q2)."""
    if rng is None:
        rng = np.random.default_rng()
    steps = len(prices1) - 1
    dt    = 1.0 / steps
    T     = 1.0
    agent = MultiAssetASAgent(gamma=gamma, rho=rho)
    x, q1, q2 = 0.0, 0, 0

    for i in range(steps):
        s1, s2 = prices1[i], prices2[i]
        t      = i * dt
        (da1, db1), (da2, db2) = agent.compute_quotes(
            [s1, s2], [q1, q2], t, T, sigma1, sigma2, k, k)
        if rng.random() < A*np.exp(-k*da1)*dt: x += s1+da1; q1 -= 1
        if rng.random() < A*np.exp(-k*db1)*dt: x -= s1-db1; q1 += 1
        if rng.random() < A*np.exp(-k*da2)*dt: x += s2+da2; q2 -= 1
        if rng.random() < A*np.exp(-k*db2)*dt: x -= s2-db2; q2 += 1

    return x + q1*prices1[-1] + q2*prices2[-1], q1, q2


def run_mc(prices1, prices2, rho, sigma1, sigma2,
           n_paths=300, seed=42):
    """Run Monte Carlo. Returns (mean_pnl, std_pnl, std_q1, std_q2)."""
    rng = np.random.default_rng(seed)
    pnls, qs1, qs2 = [], [], []
    for _ in range(n_paths):
        pnl, q1, q2 = simulate_one_path(
            prices1, prices2, rho, sigma1, sigma2, rng=rng)
        pnls.append(pnl); qs1.append(q1); qs2.append(q2)
    return (float(np.mean(pnls)), float(np.std(pnls)),
            float(np.std(qs1)),   float(np.std(qs2)))


def misspec_cost(rho_true, rho_used, sigma1, sigma2,
                 gamma=0.1, T_minus_t=0.5, q1=2, q2=-1):
    """
    Analytical misspecification cost in reservation price units.
    = difference in r1 when agent uses rho_used instead of rho_true.
    Formula: gamma*(T-t)*(rho_true-rho_used)*sigma1*sigma2*|q2|
    """
    return gamma * T_minus_t * abs(rho_true - rho_used) * sigma1 * sigma2 * abs(q2)


def make_synthetic_pair(rho_gen=0.4, sigma1=2.0, sigma2=1.5,
                         n_steps=500, seed=42):
    """Generate correlated synthetic price paths as fallback."""
    rng = np.random.default_rng(seed)
    dt  = 1.0 / n_steps
    z1  = rng.standard_normal(n_steps)
    z2  = rho_gen*z1 + np.sqrt(max(1-rho_gen**2, 0))*rng.standard_normal(n_steps)
    p1  = np.r_[100.0, 100.0 + np.cumsum(sigma1*np.sqrt(dt)*z1)]
    p2  = np.r_[ 50.0,  50.0 + np.cumsum(sigma2*np.sqrt(dt)*z2)]
    return p1, p2


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("Phase 4: Multi-Asset & Regime Extension")
    print("=" * 65)

    # ── Sanity check ────────────────────────────────────────────────────
    print("\n--- Sanity check: rho=0 collapses to single-asset formula ---")
    single  = AvellanedaStoikovAgent(gamma=0.1)
    multi_0 = MultiAssetASAgent(gamma=0.1, rho=0.0)
    sv, qv       = [100.0, 50.0], [2, -1]
    t, T, s1, s2 = 0.0, 1.0, 2.0, 1.5

    rm  = multi_0.reservation_prices(sv, qv, t, T, s1, s2)
    rs1 = single.reservation_price(100.0,  2, t, T, s1)
    rs2 = single.reservation_price( 50.0, -1, t, T, s2)
    print(f"  Asset 1: multi={rm[0]:.4f}  single={rs1:.4f}  "
          f"{'PASS' if np.isclose(rm[0],rs1) else 'FAIL'}")
    print(f"  Asset 2: multi={rm[1]:.4f}  single={rs2:.4f}  "
          f"{'PASS' if np.isclose(rm[1],rs2) else 'FAIL'}")

    # ── Load real data: build BOTH pairs ────────────────────────────────
    print("\n--- Building two stock pairs from real LOBSTER data ---")
    print("  Pair 1: AAPL vs AMZN    (different stocks, moderate correlation)")
    print("  Pair 2: AAPL morning vs AAPL afternoon  (same stock, low correlation)")
    print()

    aapl_dir, aapl_date = find_lobster_folder("AAPL")
    amzn_dir, amzn_date = find_lobster_folder("AMZN")
    use_real = (aapl_dir is not None) and (amzn_dir is not None)

    if use_real:
        try:
            mid_aapl = load_mid_prices(aapl_dir, "AAPL", aapl_date)
            mid_amzn = load_mid_prices(amzn_dir, "AMZN", amzn_date)

            # Pair 1: AAPL vs AMZN — full day
            p1_pair1, p2_pair1 = align_mid_prices(
                mid_aapl, mid_amzn, n_points=500)
            ret1_p1 = np.diff(p1_pair1) / p1_pair1[:-1]
            ret2_p1 = np.diff(p2_pair1) / p2_pair1[:-1]
            rho_pair1  = float(np.corrcoef(ret1_p1, ret2_p1)[0, 1])
            n1 = len(p1_pair1) - 1
            sig1_pair1 = float(np.std(np.diff(p1_pair1)) * np.sqrt(n1))
            sig2_pair1 = float(np.std(np.diff(p2_pair1)) * np.sqrt(n1))

            # Pair 2: AAPL morning vs AAPL afternoon
            mid_morn, mid_aft = split_intraday(mid_aapl)
            # Resample both halves to 250 points each, then align lengths
            n2 = 250
            grid_m = np.linspace(mid_morn.index[0], mid_morn.index[-1], n2)
            grid_a = np.linspace(mid_aft.index[0],  mid_aft.index[-1],  n2)

            def snap(series, grid):
                times = series.index.values; prices = series.values
                out   = np.zeros(len(grid))
                for i, t in enumerate(grid):
                    idx = max(np.searchsorted(times, t, side='right')-1, 0)
                    out[i] = prices[idx]
                return out

            p1_pair2 = snap(mid_morn, grid_m)
            p2_pair2 = snap(mid_aft,  grid_a)
            # Normalize both to start at 100 for fair comparison
            p1_pair2 = p1_pair2 / p1_pair2[0] * 100
            p2_pair2 = p2_pair2 / p2_pair2[0] * 100

            ret1_p2 = np.diff(p1_pair2) / p1_pair2[:-1]
            ret2_p2 = np.diff(p2_pair2) / p2_pair2[:-1]
            rho_pair2  = float(np.corrcoef(ret1_p2, ret2_p2)[0, 1])
            n2s = len(p1_pair2) - 1
            sig1_pair2 = float(np.std(np.diff(p1_pair2)) * np.sqrt(n2s))
            sig2_pair2 = float(np.std(np.diff(p2_pair2)) * np.sqrt(n2s))

            print(f"  Pair 1 — AAPL vs AMZN (full day):")
            print(f"    Realized rho = {rho_pair1:.4f}")
            print(f"    sigma_AAPL   = {sig1_pair1:.4f}  "
                  f"sigma_AMZN = {sig2_pair1:.4f}")
            print()
            print(f"  Pair 2 — AAPL morning vs afternoon:")
            print(f"    Realized rho = {rho_pair2:.4f}")
            print(f"    sigma_morn   = {sig1_pair2:.4f}  "
                  f"sigma_aft  = {sig2_pair2:.4f}")

        except Exception as e:
            print(f"  Error loading data: {e}. Using synthetic paths.")
            use_real = False

    if not use_real:
        print("  LOBSTER folders not found — run from inside as_project_final/")
        print("  Using synthetic paths to demonstrate the two-pair comparison.")
        # Pair 1: moderate correlation (like AAPL/AMZN)
        rho_pair1  = 0.40;  sig1_pair1 = 2.0;  sig2_pair1 = 1.5
        p1_pair1, p2_pair1 = make_synthetic_pair(rho_gen=rho_pair1,
                                                   sigma1=sig1_pair1,
                                                   sigma2=sig2_pair1)
        # Pair 2: low correlation (like AAPL morning/afternoon)
        rho_pair2  = 0.12;  sig1_pair2 = 2.0;  sig2_pair2 = 2.0
        p1_pair2, p2_pair2 = make_synthetic_pair(rho_gen=rho_pair2,
                                                   sigma1=sig1_pair2,
                                                   sigma2=sig2_pair2,
                                                   seed=99)

    # ── Reservation price table (both pairs) ────────────────────────────
    print("\n--- Reservation prices across rho values ---")
    print("  (s=[100,50], q=[2,-1], t=0, T=1)")
    for pair_name, rho_real, sigma_a, sigma_b in [
        ("AAPL/AMZN",       rho_pair1, sig1_pair1, sig2_pair1),
        ("AAPL morn/aft",   rho_pair2, sig1_pair2, sig2_pair2),
    ]:
        print(f"\n  {pair_name}  (realized rho={rho_real:.3f})")
        print(f"  {'rho':>6}  {'r1':>8}  {'r2':>8}  note")
        print("  " + "-" * 42)
        for rho in sorted({-0.5, 0.0, round(rho_real,2), 0.5, 0.9}):
            agent = MultiAssetASAgent(gamma=0.1, rho=rho)
            r     = agent.reservation_prices(
                sv, qv, 0.0, 1.0, sigma_a, sigma_b)
            note  = "<-- realized" if abs(rho-round(rho_real,2)) < 0.01 else ""
            print(f"  {rho:>6.2f}  {r[0]:>8.4f}  {r[1]:>8.4f}  {note}")

    # ── Monte Carlo: rho sweep on each pair ─────────────────────────────
    print("\n--- Monte Carlo: effect of assumed rho on risk ---")
    print("  (n_paths=300 per rho value, same price paths, only rho changes)")

    for pair_name, p1, p2, rho_real, sigma_a, sigma_b in [
        ("Pair 1: AAPL/AMZN (high corr)",
         p1_pair1, p2_pair1, rho_pair1, sig1_pair1, sig2_pair1),
        ("Pair 2: AAPL morn/aft (low corr)",
         p1_pair2, p2_pair2, rho_pair2, sig1_pair2, sig2_pair2),
    ]:
        print(f"\n  {pair_name}  realized rho={rho_real:.3f}")
        print(f"  {'rho used':>9}  {'mean_pnl':>10}  {'std_pnl':>9}  "
              f"{'std_q1':>8}  {'std_q2':>8}  note")
        print("  " + "-" * 68)
        for rho in sorted({0.0, round(rho_real,2), 0.5, 0.9}):
            m, s, q1, q2 = run_mc(p1, p2, rho, sigma_a, sigma_b,
                                   n_paths=300, seed=42)
            note = "<-- correct" if abs(rho-round(rho_real,2)) < 0.01 else ""
            print(f"  {rho:>9.2f}  {m:>10.1f}  {s:>9.1f}  "
                  f"{q1:>8.1f}  {q2:>8.1f}  {note}")

    # ── Misspecification cost comparison: the key cross-pair result ──────
    print("\n--- Misspecification cost: what using rho=0 costs analytically ---")
    print("  Formula: gamma*(T-t)*|rho_true - rho_used|*sigma1*sigma2*|q2|")
    print("  (at t=0.5, q=[2,-1], gamma=0.1)\n")
    print(f"  {'Pair':>22}  {'true rho':>9}  {'used rho':>9}  "
          f"{'cost (price units)':>20}")
    print("  " + "-" * 68)

    for pair_name, rho_true, s1, s2 in [
        ("AAPL/AMZN",       rho_pair1, sig1_pair1, sig2_pair1),
        ("AAPL morn/aft",   rho_pair2, sig1_pair2, sig2_pair2),
    ]:
        for rho_used in [0.0, round(rho_true, 2)]:
            cost = misspec_cost(rho_true, rho_used, s1, s2)
            note = "(correct)" if abs(rho_used - round(rho_true,2)) < 0.01 else "(ignoring corr.)"
            print(f"  {pair_name:>22}  {rho_true:>9.3f}  {rho_used:>9.2f}  "
                  f"{cost:>18.6f}  {note}")
    
    ratio = (misspec_cost(rho_pair1, 0.0, sig1_pair1, sig2_pair1) /
             max(misspec_cost(rho_pair2, 0.0, sig1_pair2, sig2_pair2), 1e-9))
    print(f"  Key result: misspecification cost is {ratio:.1f}x larger for")
    print(f"  AAPL/AMZN (rho={rho_pair1:.2f}) than AAPL morn/aft (rho={rho_pair2:.2f}).")
    print(f"  For low-correlation pairs, ignoring rho costs little.")
    print(f"  For high-correlation pairs, the cost scales with rho*sigma1*sigma2.")

    # ── Regime detector ──────────────────────────────────────────────────
    print("\n--- Regime detector (70/30 train/test on AAPL) ---")
    split   = int(0.7 * len(p1_pair1))
    train_p = p1_pair1[:split]
    test_p  = p1_pair1[split:]
    print(f"  Training: {len(train_p)} snapshots  |  "
          f"Test: {len(test_p)} snapshots")

    det = SimpleRegimeDetector(window=30)
    det.fit_threshold(train_p, percentile=75)

    regimes = []
    for price in test_p:
        det.update(float(price))
        regimes.append(det.current_regime())

    regimes    = np.array(regimes)
    pct_stress = np.mean(regimes == 1) * 100
    print(f"  Normal: {np.sum(regimes==0)} steps ({100-pct_stress:.1f}%)")
    print(f"  Stress: {np.sum(regimes==1)} steps ({pct_stress:.1f}%)")
    print(f"  Threshold fitted on training data only — strict no-lookahead.")
