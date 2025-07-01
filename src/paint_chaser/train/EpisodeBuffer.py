# 2025 Steven Chiacchira
"""Class for storing and computing episode statistics during training."""

import gymnasium
import torch
from torch.nn import functional as F


class EpisodeBuffer:
    """
    Class for storing and computing episode statistics during training.

    Stores states, actions, rewards, predicted action log probabilities, and returns for the episode.

    """

    def __init__(
        self,
        gamma_good: float,
        gamma_bad: float,
        bad_penalty_factor: float,
        device: torch.device,
    ) -> None:
        """
        Construct an empty `EpisodeBuffer`.

        See :py:meth:`EpisodeBuffer.compute_returns` for details on `gamma_good`, `gamma_bad`, and `bad_penalty_factor`.

        :param gamma_good: the discount factor for good rewards.
        :param gamma_bad: the discount factor for bad penalties.
        :param bad_penalty_factor: scalar used for weighting bad penalties.
        :param device: device to use for returned :py:class:`torch.Tensor`s.
        """
        self._gamma_good = gamma_good
        self._gamma_bad = gamma_bad
        self._bad_penalty_factor = bad_penalty_factor
        self.device = device

        self.states: list[torch.Tensor] = []
        self.actions: list[int] = []
        self.rewards: list[float] = []
        self.log_probs: list[torch.Tensor] = []
        self.returns: list[float] = []

    def add_record(
        self,
        state: torch.Tensor,
        action: int,
        reward: torch.Tensor,
        log_prob: torch.Tensor,
    ) -> None:
        """
        Add a new recond containing a state, action, reward, and predicted action log probabilities to the buffer.

        :param state: the environment's state.
        :param action: the action taken by the model.
        :param reward: the reward tensor given by the environment. reward[0] is the good_reward, while reward[1] is the bad_reward.
        :param log_prob: the log probabilities for actions as predicted by the model.
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)

    def compute_returns(self) -> None:
        """
        Computes the returns over the whole stored episode.

        The return for state `n`, denoted `r_n` is computed as
        `r_n` = `total_good_reward_n` - `total_bad_penalty_n` * `bad_penalty_factor`
        where
        `total_good_reward_n` = `good_reward_n` + `gamma_good` * `total_good_reward_n+1`
        `total_bad_penalty_n` = `bad_penalty_n` + `gamma_bad` * `total_bad_penalty_n+1`

        and `good_reward_n` and `bad_penalty_n` are given by the environment.
        """
        returns = []
        total_good_reward = 0
        total_bad_reward = 0
        for r in reversed(self.rewards):
            total_good_reward = r[0] + self._gamma_good * total_good_reward
            total_bad_reward = r[1] + self._gamma_bad * total_bad_reward

            total_reward = (
                total_good_reward - total_bad_reward * self._bad_penalty_factor
            )

            returns.insert(0, total_reward)
        self.returns = returns

    def get_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns a tuple of tensors containing environment states, log probabilities, and returns.

        :returns: a tuple contianing the states, log probabilities, and returns collected during the episode.
        """
        states = torch.stack(self.states)
        log_probs = torch.stack(self.log_probs).to(self.device)
        returns = torch.tensor(self.returns, dtype=torch.float32).to(self.device)

        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return states, log_probs, returns

    def collect_episode(self, agent: torch.nn.Module, env: gymnasium.Env) -> None:
        """
        Runs a single training episode for `agent` using `env`.

        :param agent: the neural network to be trained.
        :param env: the environment to train `agent` with.
        """
        state, _ = env.reset()

        max_steps_per_episode = 100
        for _step in range(max_steps_per_episode):
            state_tensor = torch.as_tensor(state, dtype=torch.float32).to(self.device)
            action_logits = agent(state_tensor.unsqueeze(0))
            action_probs = F.softmax(action_logits, dim=-1)

            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            self.add_record(state_tensor, action.item(), reward.detach(), log_prob)

            state = next_state

            if terminated or truncated:
                break

        self.compute_returns()
