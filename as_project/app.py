# python -m pip install streamlit
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from market_sim import MarketSimulator
from agent import AvellanedaStoikovAgent, SymmetricAgent
from backtest_engine import BacktestEngine

st.set_page_config(
    page_title="A-S Market Making Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Avellaneda-Stoikov (2008) — Market Making System")
st.caption("Replication, extension, and stress-testing pipeline")

# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.header("Model parameters")
gamma   = st.sidebar.slider("Risk aversion γ",  0.01, 2.0,  0.10, 0.01)
sigma   = st.sidebar.slider("Volatility σ",      0.5,  5.0,  2.00, 0.10)
k       = st.sidebar.slider("Intensity decay k", 0.5,  4.0,  1.50, 0.10)
A       = st.sidebar.slider("Arrival rate A",    50,   300,  140,  10)
n_paths = st.sidebar.selectbox("MC paths", [100, 500, 1000], index=2)
seed    = st.sidebar.number_input("Random seed", value=42, step=1)

run_btn = st.sidebar.button("Run simulation", type="primary")

# ── Tabs ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Simulation", "Backtest", "Sensitivity analysis", "Risk metrics"
])

def run_mc(gamma, sigma, A, k, n_paths, seed):
    sim   = MarketSimulator(sigma=sigma, A=A, k=k)
    agent = AvellanedaStoikovAgent(gamma=gamma)
    sym_spread = agent.optimal_spread(0, 1, sigma, k)
    sym   = SymmetricAgent(half_spread=sym_spread / 2)
    rng   = np.random.default_rng(int(seed))
    rng2  = np.random.default_rng(int(seed))

    inv_pnls, inv_qs, inv_wealth = [], [], []
    sym_pnls, sym_qs             = [], []

    for _ in range(n_paths):
        r = sim.simulate_path(agent, rng)
        inv_pnls.append(r["pnl"])
        inv_qs.append(r["final_q"])
        inv_wealth.append(r["wealth"])

    for _ in range(n_paths):
        r = sim.simulate_path(sym, rng2)
        sym_pnls.append(r["pnl"])
        sym_qs.append(r["final_q"])

    return (np.array(inv_pnls), np.array(inv_qs), inv_wealth,
            np.array(sym_pnls), np.array(sym_qs))


if run_btn:
    with st.spinner("Running Monte Carlo simulation…"):
        inv_pnls, inv_qs, inv_wealth, sym_pnls, sym_qs = run_mc(
            gamma, sigma, A, k, n_paths, seed
        )

    # ── Tab 1: Simulation results ──────────────────────────────────
    with tab1:
        st.subheader("Monte Carlo results")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Inventory mean P&L",  f"{np.mean(inv_pnls):.2f}")
        col2.metric("Inventory P&L std",   f"{np.std(inv_pnls):.2f}")
        col3.metric("Symmetric mean P&L",  f"{np.mean(sym_pnls):.2f}")
        col4.metric("Symmetric P&L std",   f"{np.std(sym_pnls):.2f}")

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].hist(inv_pnls, bins=40, alpha=0.7, label="Inventory",  edgecolor="white")
        axes[0].hist(sym_pnls, bins=40, alpha=0.7, label="Symmetric",  edgecolor="white")
        axes[0].set_title("P&L distribution")
        axes[0].set_xlabel("P&L ($)")
        axes[0].legend()

        axes[1].hist(inv_qs,  bins=30, alpha=0.7, label="Inventory",   edgecolor="white")
        axes[1].hist(sym_qs,  bins=30, alpha=0.7, label="Symmetric",   edgecolor="white")
        axes[1].set_title("Final inventory")
        axes[1].set_xlabel("Shares")
        axes[1].legend()

        # Sample wealth path
        axes[2].plot(inv_wealth[0], linewidth=1, label="Wealth path")
        axes[2].set_title("Sample wealth path")
        axes[2].set_xlabel("Time step")
        axes[2].legend()

        st.pyplot(fig)
        plt.close()

        # Paper comparison table
        st.subheader("Comparison with paper (γ=0.1)")
        comp = pd.DataFrame({
            "Strategy":  ["Inventory (this run)", "Inventory (paper)", "Symmetric (this run)", "Symmetric (paper)"],
            "Mean P&L":  [round(np.mean(inv_pnls), 1), 65.0,  round(np.mean(sym_pnls), 1), 68.4],
            "Std P&L":   [round(np.std(inv_pnls),  1),  6.6,  round(np.std(sym_pnls),  1), 12.7],
            "Mean |q|":  [round(np.mean(np.abs(inv_qs)), 2), 0.08,
                          round(np.mean(np.abs(sym_qs)), 2), 0.26],
        })
        st.dataframe(comp, use_container_width=True, hide_index=True)

    # ── Tab 2: Backtest ────────────────────────────────────────────
    with tab2:
        st.subheader("Backtest (train 70% / test 30%)")
        st.info("Agent parameters are fixed from training split. Test prices are generated fresh.")

        rng_bt   = np.random.default_rng(int(seed) + 99)
        n_bt     = 2000
        bt_prices = (100 + np.cumsum(
            rng_bt.normal(0, sigma * np.sqrt(0.005), n_bt)
        ))

        engine = BacktestEngine(train_ratio=0.7)
        agent_bt = AvellanedaStoikovAgent(gamma=gamma)
        bt = engine.run(bt_prices, agent_bt,
                        {"sigma": sigma, "k": k, "A": A, "dt": 0.005})

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Final P&L",    f"{bt.final_pnl:.2f}")
        c2.metric("Sharpe ratio", f"{bt.sharpe:.3f}")
        c3.metric("Max drawdown", f"{bt.max_drawdown:.2f}")
        c4.metric("Total trades", bt.total_trades)

        fig2, axes2 = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        axes2[0].plot(bt.price_series,     linewidth=0.8, label="Mid-price")
        axes2[0].set_ylabel("Price ($)")
        axes2[0].legend()
        axes2[1].plot(bt.pnl_series,       linewidth=1,   color="tab:green", label="P&L")
        axes2[1].set_ylabel("P&L ($)")
        axes2[1].legend()
        axes2[2].plot(bt.inventory_series, linewidth=1,   color="tab:orange", label="Inventory q")
        axes2[2].axhline(0, linewidth=0.5, linestyle="--", color="gray")
        axes2[2].set_ylabel("Shares")
        axes2[2].set_xlabel("Test step")
        axes2[2].legend()
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close()

    # ── Tab 3: Sensitivity ─────────────────────────────────────────
    with tab3:
        st.subheader("Sensitivity analysis — γ sweep")
        gamma_vals = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
        rows = []
        prog = st.progress(0)
        for idx, g in enumerate(gamma_vals):
            ip, iq, _, sp, sq = run_mc(g, sigma, A, k, 300, seed)
            rows.append({
                "γ":              g,
                "Mean P&L":       round(float(np.mean(ip)), 2),
                "Std P&L":        round(float(np.std(ip)),  2),
                "Mean |q| final": round(float(np.mean(np.abs(iq))), 2),
                "Avg spread":     round(AvellanedaStoikovAgent(g)
                                         .optimal_spread(0, 1, sigma, k), 3),
            })
            prog.progress((idx + 1) / len(gamma_vals))

        df_sens = pd.DataFrame(rows)
        st.dataframe(df_sens, use_container_width=True, hide_index=True)

        fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4))
        axes3[0].plot(df_sens["γ"], df_sens["Mean P&L"],  marker="o", label="Mean P&L")
        axes3[0].plot(df_sens["γ"], df_sens["Std P&L"],   marker="s", label="Std P&L")
        axes3[0].set_xlabel("γ")
        axes3[0].set_ylabel("$")
        axes3[0].set_title("P&L vs risk aversion")
        axes3[0].legend()
        axes3[1].plot(df_sens["γ"], df_sens["Avg spread"],   marker="o", color="tab:red")
        axes3[1].set_xlabel("γ")
        axes3[1].set_ylabel("Spread ($)")
        axes3[1].set_title("Optimal spread vs γ")
        st.pyplot(fig3)
        plt.close()

    # ── Tab 4: Risk metrics ────────────────────────────────────────
    with tab4:
        st.subheader("Risk diagnostics")

        sharpe_inv = (np.mean(inv_pnls) / np.std(inv_pnls)
                      if np.std(inv_pnls) > 0 else 0)
        sharpe_sym = (np.mean(sym_pnls) / np.std(sym_pnls)
                      if np.std(sym_pnls) > 0 else 0)
        var_95_inv  = float(np.percentile(inv_pnls, 5))
        var_95_sym  = float(np.percentile(sym_pnls, 5))

        risk_df = pd.DataFrame({
            "Metric": ["Sharpe (daily)", "VaR 95%", "Max |inventory|",
                       "Std P&L", "Pct paths q=0 at T"],
            "Inventory": [
                round(sharpe_inv, 3),
                round(var_95_inv, 2),
                int(np.max(np.abs(inv_qs))),
                round(float(np.std(inv_pnls)), 2),
                round(float(np.mean(inv_qs == 0)) * 100, 1),
            ],
            "Symmetric": [
                round(sharpe_sym, 3),
                round(var_95_sym, 2),
                int(np.max(np.abs(sym_qs))),
                round(float(np.std(sym_pnls)), 2),
                round(float(np.mean(sym_qs == 0)) * 100, 1),
            ],
        })
        st.dataframe(risk_df, use_container_width=True, hide_index=True)

        fig4, axes4 = plt.subplots(1, 2, figsize=(12, 4))
        axes4[0].hist(inv_pnls, bins=40, alpha=0.7, label="Inventory", edgecolor="white")
        axes4[0].axvline(var_95_inv, color="red",    linestyle="--", label=f"VaR 95% = {var_95_inv:.1f}")
        axes4[0].set_title("Inventory strategy P&L + VaR")
        axes4[0].legend()
        axes4[1].hist(sym_pnls, bins=40, alpha=0.7, label="Symmetric",
                      color="tab:orange", edgecolor="white")
        axes4[1].axvline(var_95_sym, color="red",    linestyle="--", label=f"VaR 95% = {var_95_sym:.1f}")
        axes4[1].set_title("Symmetric strategy P&L + VaR")
        axes4[1].legend()
        st.pyplot(fig4)
        plt.close()

else:
    for tab in [tab1, tab2, tab3, tab4]:
        with tab:
            st.info("Set parameters in the sidebar and click **Run simulation** to start.")
