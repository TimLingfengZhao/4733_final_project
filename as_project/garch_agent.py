import numpy as np
from agent import AvellanedaStoikovAgent


class GARCHPriceSimulator:
    """
    Generates price paths with GARCH(1,1) volatility clustering.
    Parameters chosen so unconditional vol matches paper's sigma=2:
        omega/(1-alpha-beta) = (sigma*sqrt(dt))^2 = 0.02
    """
    def __init__(self, S0=100, dt=0.005, T=1.0,
                 omega=0.0010, alpha=0.25, beta=0.70):
        self.S0=S0; self.dt=dt; self.T=T
        self.steps=int(T/dt)
        self.omega=omega; self.alpha=alpha; self.beta=beta

    def simulate_path(self, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        sigma2 = self.omega / (1 - self.alpha - self.beta)
        S = self.S0; prices=[S]; vols=[]
        for _ in range(self.steps):
            st = np.sqrt(sigma2)
            ret = st * rng.standard_normal()
            S += ret; prices.append(S); vols.append(st)
            sigma2 = self.omega + self.alpha*ret**2 + self.beta*sigma2
        return np.array(prices), np.array(vols)


class GARCHAdaptiveAgent:
    """
    A-S agent with rolling vol estimate.
    Uses rolling std of past 80 price returns, rescaled to per-unit-time sigma.
    Simple and fast — no full GARCH MLE each step (too slow for 1000 paths).
    The key fix vs the original: divide by sqrt(dt) to convert raw return std
    back to the per-unit-time sigma that the A-S spread formula expects.
    """
    def __init__(self, gamma=0.1, dt=0.005, window=80):
        self.base = AvellanedaStoikovAgent(gamma=gamma)
        self.dt = dt; self.window = window
        self.price_history = []

    def compute_quotes(self, s, q, t, T, sigma_ignored, k):
        self.price_history.append(s)
        hist = self.price_history[-self.window:]
        if len(hist) > 5:
            rets = np.diff(hist)
            raw_vol = float(np.std(rets))
            # Convert raw return std -> per-unit-time sigma
            sigma_t = np.clip(raw_vol / np.sqrt(self.dt), 0.5, 8.0)
        else:
            sigma_t = 2.0   # warm-up: use paper default
        return self.base.compute_quotes(s, q, t, T, sigma_t, k)


def run_phase2(n_paths=1000, gamma=0.1, seed=42):
    sim  = GARCHPriceSimulator()
    A, k = 140, 1.5

    print("=" * 60)
    print("Phase 2: GARCH(1,1) Stress Test")
    print("Price process: GARCH(1,1) clustering (same unconditional vol as paper)")
    print("=" * 60)

    def run_agent(make_agent, rng_seed):
        rng  = np.random.default_rng(rng_seed)
        pnls, qs = [], []
        for _ in range(n_paths):
            prices, _ = sim.simulate_path(rng)
            agent = make_agent(); x=0.0; q=0
            for i in range(sim.steps):
                S=prices[i]; t=i*sim.dt
                d_a, d_b = agent.compute_quotes(S, q, t, sim.T, 2.0, k)
                pa=S+d_a; pb=S-d_b
                if rng.random() < A*np.exp(-k*d_a)*sim.dt: x+=pa; q-=1
                if rng.random() < A*np.exp(-k*d_b)*sim.dt: x-=pb; q+=1
            pnls.append(x + q*prices[-1]); qs.append(q)
        return np.mean(pnls), np.std(pnls), np.std(qs)

    m1,s1,q1 = run_agent(lambda: AvellanedaStoikovAgent(gamma=gamma), seed)
    m2,s2,q2 = run_agent(lambda: GARCHAdaptiveAgent(gamma=gamma), seed)

    print(f"   AS_fixed_sigma:    pnl={m1:.1f}  std={s1:.1f}  std_q={q1:.1f}")
    print(f"   AS_garch_adaptive: pnl={m2:.1f}  std={s2:.1f}  std_q={q2:.1f}")
    print()
    if s2 < s1:
        print(f"  GARCH agent reduced P&L std by {(1-s2/s1)*100:.1f}% — vol adaptation works")
    else:
        print(f"  Fixed agent still more stable — consider increasing GARCH window")


if __name__ == "__main__":
    run_phase2()
