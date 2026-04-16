# Avellaneda-Stoikov (2008) — Replication Project

Replication, extension, and stress-testing of:
> Avellaneda & Stoikov, "High-frequency trading in a limit order book",
> *Quantitative Finance*, Vol. 8 No. 3, 2008.

---

## Project structure

```
as_project/
├── market_sim.py          # Phase 1 — price process & Poisson order arrivals
├── agent.py               # Phase 1 — reservation price & optimal spread
├── run_phase1.py          # Phase 1 — reproduce Tables 1–3 from the paper
├── garch_vol.py           # Phase 2 — rolling GARCH(1,1) vol estimator
├── garch_agent.py         # Phase 2 — GARCH-adaptive agent & stress test
├── lobster_loader.py      # Phase 3 — LOBSTER tick data parser & λ(δ) calibration
├── multi_asset_agent.py   # Phase 4 — 2-asset correlated agent & regime detector
├── backtest_engine.py     # Phase 5 — strict no-lookahead backtest engine
├── app.py                 # Phase 5 — Streamlit dashboard (main deliverable)
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Usage

### Phase 1 — Reproduce paper Tables 1–3
```bash
python run_phase1.py
```
Expected output (γ=0.1):
- Inventory: pnl ≈ 65.0, std ≈ 6.6, final_q ≈ 0.08
- Symmetric: pnl ≈ 68.4, std ≈ 12.7, final_q ≈ 0.26

### Phase 2 — GARCH stress test
```bash
python garch_agent.py
```

### Phase 5 — Full deployed dashboard
```bash
streamlit run app.py
```
Open http://localhost:8501 in your browser.
Set parameters in the sidebar and click **Run simulation**.

---

## Key formulas

**Reservation price:**
```
r(s, q, t) = s − q · γ · σ² · (T − t)
```

**Optimal bid-ask spread:**
```
δᵃ + δᵇ = γσ²(T−t) + (2/γ)·ln(1 + γ/k)
```

**Order arrival intensity:**
```
λ(δ) = A · exp(−k · δ)
```

---

## Validation checklist

- [ ] Phase 1: P&L std for inventory ~6.6 vs symmetric ~12.7 (γ=0.1)
- [ ] Phase 2: GARCH agent partially recovers risk reduction under vol clustering
- [ ] Phase 3: Exponential fit R² > 0.85 on LOBSTER data
- [ ] Phase 4: Multi-asset r_i collapses to single-asset at ρ=0
- [ ] Phase 5: App runs, train/test split enforced, sensitivity table correct

---

## No-lookahead guarantee

`BacktestEngine` enforces a strict 70/30 train/test split.
All parameter fitting (GARCH thresholds, calibrated k and A) must be
done on `price_data[:split]` before `engine.run()` is called.
The test loop executes at price[t] and fills at price[t+1] to simulate latency.
