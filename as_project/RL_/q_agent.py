import numpy as np


class QLearningAgent:
    """
    Tabular Q-learning market-making agent.

    State buckets:
      - inventory q: [-q_clip, q_clip]
      - time-to-maturity: t_buckets
      - realized volatility regime: low/medium/high

    Action space:
      - (delta_a, delta_b) from a small discrete grid
    """

    def __init__(
        self,
        q_clip=5,
        t_buckets=10,
        dt=0.005,
        vol_window=20,
        action_grid=None,
        alpha=0.1,
        discount=0.95,
        epsilon=0.30,
        epsilon_min=0.02,
        epsilon_decay=0.999,
        inventory_penalty=0.01,
    ):
        self.q_clip = int(q_clip)
        self.t_buckets = int(t_buckets)
        self.dt = float(dt)
        self.vol_window = int(vol_window)

        self.alpha = float(alpha)
        self.discount = float(discount)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.inventory_penalty = float(inventory_penalty)

        self.action_grid = np.array(
            action_grid if action_grid is not None else [0.10, 0.30, 0.50, 0.80, 1.20],
            dtype=float,
        )
        self.actions = [(da, db) for da in self.action_grid for db in self.action_grid]
        self.n_actions = len(self.actions)

        n_q = 2 * self.q_clip + 1
        self.q_table = np.zeros((n_q, self.t_buckets, 3, self.n_actions), dtype=float)
        self._price_history = []

    def reset_episode(self):
        self._price_history = []

    def _clip_q(self, q):
        return int(np.clip(q, -self.q_clip, self.q_clip))

    def _q_idx(self, q):
        return self._clip_q(q) + self.q_clip

    def _t_idx(self, t, T):
        if T <= 0:
            return 0
        frac = np.clip(t / T, 0.0, 0.999999)
        return int(frac * self.t_buckets)

    def _vol_bucket(self, sigma, price_history):
        # Compare realized step vol against model-implied step vol.
        if len(price_history) < 3:
            return 1
        rets = np.diff(price_history[-self.vol_window :])
        realized = float(np.std(rets)) if len(rets) > 1 else 0.0
        expected = max(float(sigma) * np.sqrt(self.dt), 1e-8)
        ratio = realized / expected
        if ratio < 0.75:
            return 0
        if ratio < 1.25:
            return 1
        return 2

    def build_state(self, q, t, T, sigma, price_history):
        return (
            self._q_idx(q),
            self._t_idx(t, T),
            self._vol_bucket(sigma, price_history),
        )

    def select_action(self, state, training=True, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        q_idx, t_idx, v_idx = state
        if training and rng.random() < self.epsilon:
            action_idx = int(rng.integers(0, self.n_actions))
        else:
            action_idx = int(np.argmax(self.q_table[q_idx, t_idx, v_idx]))
        return action_idx, self.actions[action_idx]

    def update(self, state, action_idx, reward, next_state):
        q_idx, t_idx, v_idx = state
        nq_idx, nt_idx, nv_idx = next_state
        q_sa = self.q_table[q_idx, t_idx, v_idx, action_idx]
        td_target = reward + self.discount * np.max(self.q_table[nq_idx, nt_idx, nv_idx])
        self.q_table[q_idx, t_idx, v_idx, action_idx] = q_sa + self.alpha * (td_target - q_sa)

    def end_episode(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def compute_quotes(self, s, q, t, T, sigma, k):
        # k is part of shared strategy API; tabular policy already internalizes it during training.
        _ = s, k
        self._price_history.append(float(s))
        state = self.build_state(q, t, T, sigma, self._price_history)
        action_idx, (delta_a, delta_b) = self.select_action(state, training=False)
        _ = action_idx
        return max(delta_a, 1e-4), max(delta_b, 1e-4)

    def save(self, path):
        np.save(path, self.q_table)

    def load(self, path):
        self.q_table = np.load(path)

