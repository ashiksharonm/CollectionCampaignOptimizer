import random
from abc import ABC, abstractmethod

import numpy as np

from src.features.schemas import Channel


class BasePolicy(ABC):
    def __init__(self, n_actions: int, feature_dim: int):
        self.n_actions = n_actions
        self.feature_dim = feature_dim
        self.channels = list(Channel)
        self.t = 0

    @abstractmethod
    def select_action(self, context: np.ndarray) -> Channel:
        pass

    @abstractmethod
    def update(self, context: np.ndarray, action: Channel, reward: float):
        pass

    def get_model_name(self) -> str:
        return self.__class__.__name__


class RandomPolicy(BasePolicy):
    def select_action(self, context: np.ndarray) -> Channel:
        return random.choice(self.channels)

    def update(self, context: np.ndarray, action: Channel, reward: float):
        pass


class EpsilonGreedyPolicy(BasePolicy):
    """
    Context-free Epsilon Greedy.
    Maintains mean reward for each arm.
    """

    def __init__(self, n_actions: int, feature_dim: int, epsilon: float = 0.1):
        super().__init__(n_actions, feature_dim)
        self.epsilon = epsilon
        self.counts = np.zeros(n_actions)
        self.values = np.zeros(n_actions)  # Average reward

    def select_action(self, context: np.ndarray) -> Channel:
        if random.random() < self.epsilon:
            return random.choice(self.channels)

        # Greedy
        action_idx = np.argmax(self.values)
        return self.channels[action_idx]

    def update(self, context: np.ndarray, action: Channel, reward: float):
        idx = self.channels.index(action)
        self.counts[idx] += 1
        n = self.counts[idx]
        # Incremental mean update
        value = self.values[idx]
        new_value = value + (reward - value) / n
        self.values[idx] = new_value


class LinUCBPolicy(BasePolicy):
    """
    Disjoint LinUCB. Each arm has its own Ridge Regression.
    """

    def __init__(self, n_actions: int, feature_dim: int, alpha: float = 1.0):
        super().__init__(n_actions, feature_dim)
        self.alpha = alpha

        # A_inv: (n_actions, dim, dim) - store inverse covariance directly or invert on fly
        # Storing A and b usually.
        # A: (dim, dim) + Identity
        self.A = [np.identity(feature_dim) for _ in range(n_actions)]
        self.b = [np.zeros(feature_dim) for _ in range(n_actions)]

    def select_action(self, context: np.ndarray) -> Channel:
        # p_ta = theta_a.T * x_t + alpha * sqrt(x_t.T * A_a_inv * x_t)
        p_t = np.zeros(self.n_actions)

        for a in range(self.n_actions):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]

            mean = theta @ context
            var = context @ A_inv @ context
            ucb = self.alpha * np.sqrt(var)

            p_t[a] = mean + ucb

        action_idx = np.argmax(p_t)
        return self.channels[action_idx]

    def update(self, context: np.ndarray, action: Channel, reward: float):
        a = self.channels.index(action)
        self.A[a] += np.outer(context, context)
        self.b[a] += reward * context


class ThompsonSamplingPolicy(BasePolicy):
    """
    Gaussian Thompson Sampling with linear payoff (Bayesian Linear Regression).
    Prior: theta ~ N(0, I)
    Likelihood: y ~ N(theta.T * x, sigma^2)
    Posterior is Gaussian.
    """

    def __init__(
        self, n_actions: int, feature_dim: int, sigma: float = 1.0, v: float = 1.0
    ):
        super().__init__(n_actions, feature_dim)
        self.v = v  # Scaling factor for variance

        self.B = [
            np.identity(feature_dim) for _ in range(n_actions)
        ]  # Precision matrix (inverse covariance)
        self.mu_hat = [np.zeros(feature_dim) for _ in range(n_actions)]
        self.f = [np.zeros(feature_dim) for _ in range(n_actions)]  # Accumulated x*y

    def select_action(self, context: np.ndarray) -> Channel:
        sampled_means = np.zeros(self.n_actions)

        for a in range(self.n_actions):
            # Covariance = B_inv * v^2
            B_inv = np.linalg.inv(self.B[a])
            mean = B_inv @ self.f[a]
            cov = B_inv * (self.v**2)

            # Sample theta from posterior
            theta_sample = np.random.multivariate_normal(mean, cov)
            sampled_means[a] = theta_sample @ context

        action_idx = np.argmax(sampled_means)
        return self.channels[action_idx]

    def update(self, context: np.ndarray, action: Channel, reward: float):
        a = self.channels.index(action)
        self.B[a] += np.outer(context, context)
        self.f[a] += reward * context
