import numpy as np


class AvellanedaStoikovAgent:
    """
    Reservation (indifference) price:
        r(s,q,t) = s - q * gamma * sigma^2 * (T - t)

    Optimal bid/ask spread (approximate closed-form):
        delta_a + delta_b = gamma*sigma^2*(T-t) + (2/gamma)*ln(1 + gamma/k)

    Quotes are centred on r, not on mid-price s.
    """

    def __init__(self, gamma=0.1):
        self.gamma = gamma

    def reservation_price(self, s, q, t, T, sigma):
        return s - q * self.gamma * sigma ** 2 * (T - t)

    def optimal_spread(self, t, T, sigma, k):
        gamma = self.gamma
        term1 = gamma * sigma ** 2 * (T - t)
        term2 = (2 / gamma) * np.log(1 + gamma / k)
        return term1 + term2

    def compute_quotes(self, s, q, t, T, sigma, k):
        """
        Returns (delta_a, delta_b) — distances from mid to ask/bid.
        Both quotes are centred on reservation price r, not mid s.
        """
        r      = self.reservation_price(s, q, t, T, sigma)
        spread = self.optimal_spread(t, T, sigma, k)

        half_spread = spread / 2
        delta_a = (r - s) + half_spread   # ask distance from mid
        delta_b = (s - r) + half_spread   # bid distance from mid

        # Clamp: quotes must be > 0 distance from mid
        delta_a = max(delta_a, 1e-4)
        delta_b = max(delta_b, 1e-4)
        return delta_a, delta_b


class SymmetricAgent:
    """Benchmark: fixed spread centred on mid-price (no inventory adjustment)."""

    def __init__(self, half_spread=0.745):
        self.half_spread = half_spread

    def compute_quotes(self, s, q, t, T, sigma, k):
        return self.half_spread, self.half_spread
