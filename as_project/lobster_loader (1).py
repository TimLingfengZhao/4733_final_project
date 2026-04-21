import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit


class LobsterLoader:
    """
    LOBSTER format:
      orderbook file:  [ask1, asksize1, bid1, bidsize1, ask2, ...]
      message file:    [time, type, orderid, size, price, direction]
    Message types: 1=new, 2=cancel, 3=delete, 4=exec_visible,
                   5=exec_hidden, 7=halt
    """

    def __init__(self, data_dir: str, levels: int = 10):
        self.data_dir = Path(data_dir)
        self.levels   = levels

    def load_day(self, ticker: str, date: str):
        """Load one stock-day. date format: 'YYYY-MM-DD'."""
        msg_file = (
            self.data_dir
            / f"{ticker}_{date}_34200000_57600000_message_{self.levels}.csv"
        )
        lob_file = (
            self.data_dir
            / f"{ticker}_{date}_34200000_57600000_orderbook_{self.levels}.csv"
        )

        msg_cols = ["time", "type", "order_id", "size", "price", "direction"]
        lob_cols = []
        for i in range(1, self.levels + 1):
            lob_cols += [
                f"ask_price_{i}", f"ask_size_{i}",
                f"bid_price_{i}", f"bid_size_{i}",
            ]

        messages  = pd.read_csv(msg_file, header=None, names=msg_cols)
        orderbook = pd.read_csv(lob_file, header=None, names=lob_cols)

        # LOBSTER prices are in integer units of 1/10000 dollar
        messages["price"] /= 10000
        price_cols = [c for c in lob_cols if "price" in c]
        orderbook[price_cols] /= 10000

        return messages, orderbook

    def compute_mid_price(self, orderbook: pd.DataFrame) -> pd.Series:
        return (orderbook["ask_price_1"] + orderbook["bid_price_1"]) / 2

    def extract_executions(self, messages: pd.DataFrame,
                           orderbook: pd.DataFrame):
        """
        Extract all type-4/5 executions with their delta from mid.

        Delta = |execution_price - mid_price_at_execution_time|

        For AAPL (penny-spread stock), most executions are at the
        best quote (delta = half-spread ~ $0.05). Large market orders
        that walk the book execute at deeper levels (larger delta).
        Both types are included — this gives the full lambda(delta) curve.
        """
        exec_mask   = messages["type"].isin([4, 5])
        execs       = messages[exec_mask].copy()
        lob_at_exec = orderbook.loc[execs.index]
        mid         = self.compute_mid_price(lob_at_exec)

        execs["mid"]   = mid.values
        execs["delta"] = np.abs(execs["price"] - mid.values)
        execs = execs[execs["delta"] > 0]
        return execs


class ArrivalRateCalibrator:
    """
    Estimates lambda(delta) from execution data and fits:
      1. Exponential: lambda(d) = A * exp(-k*d)   [A-S paper]
      2. Power-law:   lambda(d) = B * d^(-alpha)  [alternative]

    Uses log-spaced bins to capture the full range from near-mid
    (frequent small executions) to far-from-mid (rare large orders).
    """

    def __init__(self, n_bins=25):
        self.n_bins = n_bins

    def empirical_intensity(self, deltas: np.ndarray, T_total: float):
        """
        Bin deltas with log spacing and compute arrival rate per bin.

        Why log spacing: delta values span 2-3 orders of magnitude
        (e.g. 0.01 to 2.00). Linear bins would put almost all
        executions in the first bin. Log spacing gives even coverage
        across the full range and produces a clean exponential decay
        curve when plotted.
        """
        deltas = deltas[deltas > 0]

        d_min = np.percentile(deltas, 1)    # ignore extreme outliers
        d_max = np.percentile(deltas, 99)

        if d_min <= 0:
            d_min = deltas[deltas > 0].min()

        # Log-spaced bin edges
        edges       = np.logspace(np.log10(d_min), np.log10(d_max),
                                  self.n_bins + 1)
        counts, _   = np.histogram(deltas, bins=edges)
        bin_width   = np.diff(edges)
        bin_centres = np.sqrt(edges[:-1] * edges[1:])  # geometric mean

        # Rate = count / (T_total * bin_width)  [arrivals per second per unit delta]
        rates = counts / (T_total * bin_width)
        mask  = (rates > 0) & (bin_centres > 0)
        return bin_centres[mask], rates[mask]

    def fit_exponential(self, delta_vals: np.ndarray,
                        rates: np.ndarray):
        """Fit A*exp(-k*delta). Returns (A, k, r_squared)."""
        def model(d, A, k):
            return A * np.exp(-k * d)

        # Initial guess from data
        A_guess = float(np.max(rates))
        k_guess = float(1.0 / np.median(delta_vals))

        try:
            popt, _ = curve_fit(
                model, delta_vals, rates,
                p0=[A_guess, k_guess],
                bounds=([0, 0.01], [np.inf, 1000]),
                maxfev=20000
            )
            fitted = model(delta_vals, *popt)
            ss_res = np.sum((rates - fitted) ** 2)
            ss_tot = np.sum((rates - np.mean(rates)) ** 2)
            r2     = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
            return float(popt[0]), float(popt[1]), r2
        except (RuntimeError, ValueError):
            return None, None, None

    def fit_power_law(self, delta_vals: np.ndarray,
                      rates: np.ndarray):
        """Fit B*delta^(-alpha) in log space. Returns (B, alpha, r_squared)."""
        mask = (delta_vals > 0) & (rates > 0)
        if mask.sum() < 3:
            return None, None, None

        log_d  = np.log(delta_vals[mask])
        log_r  = np.log(rates[mask])
        coeffs = np.polyfit(log_d, log_r, 1)
        alpha  = float(-coeffs[0])
        B      = float(np.exp(coeffs[1]))

        fitted = B * delta_vals[mask] ** (-alpha)
        ss_res = np.sum((rates[mask] - fitted) ** 2)
        ss_tot = np.sum((rates[mask] - np.mean(rates[mask])) ** 2)
        r2     = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        return B, alpha, r2

    def calibrate(self, deltas: np.ndarray, T_total: float):
        """
        Full calibration pipeline.
        deltas  : raw delta values from extract_executions()
        T_total : total trading time in seconds
        """
        d_vals, rates = self.empirical_intensity(deltas, T_total)

        print(f"Delta range:      {d_vals.min():.4f}  to  {d_vals.max():.4f}")
        print(f"Rate range:       {rates.min():.2f}  to  {rates.max():.2f}")
        print(f"Data points:      {len(d_vals)}")
        print()

        A, k, r2_exp  = self.fit_exponential(d_vals, rates)
        B, alp, r2_pw = self.fit_power_law(d_vals, rates)

        def fmt(v, decimals=3):
            return f"{v:.{decimals}f}" if v is not None else "N/A"

        print(f"Exponential fit:  A={fmt(A,1)},  k={fmt(k,3)},  R^2={fmt(r2_exp,4)}")
        print(f"Power-law fit:    B={fmt(B,1)},  alpha={fmt(alp,3)},  R^2={fmt(r2_pw,4)}")
        print(f"Paper params:     A=140,   k=1.5")

        return {
            "exponential": {"A": A,   "k": k,     "r2": r2_exp},
            "power_law":   {"B": B,   "alpha": alp, "r2": r2_pw},
            "delta_vals":  d_vals,
            "rates":       rates,
        }


class TransactionCostModel:
    """
    Realistic execution costs:
      1. Fixed per-trade fee (e.g. $0.0035/share)
      2. Market impact slippage proportional to sqrt(order size)
    """

    def __init__(self, fee_per_share=0.0035, impact_coeff=0.01):
        self.fee   = fee_per_share
        self.gamma = impact_coeff

    def execution_price(self, quote_price, size, direction):
        slippage = self.gamma * np.sqrt(size)
        fee      = self.fee * size
        return quote_price + direction * slippage + direction * fee

    def adjust_pnl(self, gross_pnl, n_trades, avg_size=1):
        return gross_pnl - n_trades * self.fee * avg_size


if __name__ == "__main__":
    loader = LobsterLoader("LOBSTER_SampleFile_AAPL_2012-06-21_10", levels=10)
    messages, orderbook = loader.load_day("AAPL", "2012-06-21")
    print("Loaded successfully.")
    print("Messages shape:", messages.shape)
    print("Orderbook shape:", orderbook.shape)

    T_total = float(messages["time"].max() - messages["time"].min())
    print(f"Trading window: {T_total:.1f} seconds\n")

    # Extract executions
    execs = loader.extract_executions(messages, orderbook)
    print(f"Executions: {len(execs)}")
    print(f"Delta stats: min={execs['delta'].min():.4f}  "
          f"median={execs['delta'].median():.4f}  "
          f"max={execs['delta'].max():.4f}")
    print(f"Unique delta values: {execs['delta'].nunique()}\n")

    # Calibrate using log-spaced bins on raw deltas
    calibrator = ArrivalRateCalibrator(n_bins=25)
    results = calibrator.calibrate(execs["delta"].values, T_total)

    # Compare paper vs fitted at a reference delta
    A_fit = results["exponential"]["A"]
    k_fit = results["exponential"]["k"]
    if A_fit is not None:
        print(f"\nAt delta=0.10:")
        print(f"  Fitted model:  lambda = {A_fit * np.exp(-k_fit * 0.10):.1f}")
        print(f"  Paper model:   lambda = {140 * np.exp(-1.5 * 0.10):.1f}")
