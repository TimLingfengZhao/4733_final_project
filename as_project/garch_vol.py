import numpy as np
from scipy.optimize import minimize


class GARCHVolEstimator:
    """
    GARCH(1,1): sigma_t^2 = omega + alpha*eps_{t-1}^2 + beta*sigma_{t-1}^2
    Fitted by maximum likelihood on a rolling window of past returns.
    """

    def __init__(self, window=200):
        self.window  = window
        self._params = None  # (omega, alpha, beta)

    def _neg_log_likelihood(self, params, returns):
        omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
            return 1e10
        n      = len(returns)
        sigma2 = np.var(returns) * np.ones(n)
        ll     = 0.0
        for t in range(1, n):
            sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]
            ll += -0.5 * (
                np.log(2 * np.pi)
                + np.log(sigma2[t])
                + returns[t] ** 2 / sigma2[t]
            )
        return -ll

    def fit(self, price_history: np.ndarray):
        """Fit GARCH on returns from price_history. Call only on training data."""
        returns = np.diff(price_history)
        if len(returns) < 10:
            return
        x0     = [0.01, 0.1, 0.85]
        bounds = [(1e-6, 1), (1e-6, 0.5), (1e-6, 0.999)]
        res    = minimize(
            self._neg_log_likelihood, x0,
            args=(returns,), bounds=bounds, method="L-BFGS-B"
        )
        if res.success:
            self._params = res.x

    def current_sigma(self, price_history: np.ndarray) -> float:
        """
        Return current GARCH conditional vol estimate.
        Falls back to rolling std if GARCH not yet fitted.
        """
        if self._params is None or len(price_history) < self.window:
            returns = np.diff(price_history[-self.window:])
            return float(np.std(returns)) if len(returns) > 1 else 2.0

        omega, alpha, beta = self._params
        returns = np.diff(price_history[-self.window:])
        sigma2  = np.var(returns)
        for r in returns:
            sigma2 = omega + alpha * r ** 2 + beta * sigma2
        return float(np.sqrt(sigma2))
