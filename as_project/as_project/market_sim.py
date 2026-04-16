import numpy as np


class MarketSimulator:
    """
    Mid-price follows arithmetic Brownian motion:
        dS = sigma * dW
    Market orders arrive as Poisson processes with
    intensity lambda(delta) = A * exp(-k * delta)
    """

    def __init__(self, S0=100, sigma=2, A=140, k=1.5, dt=0.005, T=1):
        self.S0    = S0
        self.sigma = sigma
        self.A     = A
        self.k     = k
        self.dt    = dt
        self.T     = T
        self.steps = int(T / dt)

    def arrival_intensity(self, delta: float) -> float:
        """Poisson intensity for a quote at distance delta from mid."""
        return self.A * np.exp(-self.k * delta)

    def simulate_path(self, agent, rng=None):
        """
        Run one simulation path. Returns dict with:
          pnl, final_q, wealth_series, price_series
        """
        if rng is None:
            rng = np.random.default_rng()

        S = self.S0
        x = 0.0    # cash wealth
        q = 0      # inventory (integer shares)
        wealth_hist = []
        price_hist  = []

        for i in range(self.steps):
            t = i * self.dt

            # Agent computes quotes given current state (no lookahead)
            delta_a, delta_b = agent.compute_quotes(
                S, q, t, self.T, self.sigma, self.k
            )
            pa = S + delta_a  # ask price
            pb = S - delta_b  # bid price

            # Poisson arrivals in interval dt
            lam_a = self.arrival_intensity(delta_a)
            lam_b = self.arrival_intensity(delta_b)

            hit_ask = rng.random() < lam_a * self.dt  # sell executed
            hit_bid = rng.random() < lam_b * self.dt  # buy executed

            if hit_ask:
                x += pa
                q -= 1
            if hit_bid:
                x -= pb
                q += 1

            # Mid-price update: dS = sigma * sqrt(dt) * Z
            S += self.sigma * np.sqrt(self.dt) * rng.standard_normal()

            wealth_hist.append(x + q * S)
            price_hist.append(S)

        pnl = x + q * S  # mark inventory to market at T
        return {
            "pnl":    pnl,
            "final_q": q,
            "wealth": np.array(wealth_hist),
            "prices": np.array(price_hist),
        }
