from typing import Dict, Optional, Protocol, TypeVar
import gymnasium
from pettingzoo import AECEnv


Action = TypeVar("Action")
Observation = TypeVar("Observation")


class ActOther(Protocol):
  def act(observation: Observation) -> Action:
    ...


class PZ2GymnasiumWrapper(gymnasium.Env):
  """
  This class wraps a PettingZoo environment (AECEnv) and exposes a simple Gymnasium environment (for single agent).
  In order to make it work, one needs to provide in the initialization a mechanism to get the actions of all other agents.
  """

  def __init__(self, pz_env: AECEnv, act_others: Dict[str, ActOther], take_spaces_from: str = None):
    super().__init__()
    self.pz_env = pz_env
    self.act_others = act_others
    take_spaces_from = take_spaces_from or next(iter(act_others)) # just use any of the keys in the act_others dict.
    self.observation_space = self.pz_env.observation_space(take_spaces_from)
    self.action_space = self.pz_env.action_space(take_spaces_from)

  def reset(
    self,
    seed: Optional[int] = None,
    options: Optional[dict] = None,
  ):
    super().reset(seed=seed)
    self.pz_env.reset(seed=seed)
    self._loop_others()
    agent = self.pz_env.agent_selection
    observation, info = (
      self.pz_env.observe(agent),
      self.pz_env.infos[agent]
    )
    return observation, info

  def step(self, action):
    agent = self.pz_env.agent_selection
    assert agent not in self.act_others, f"expected it to be my turn, got {agent}"
    self.pz_env.step(action)
    # agent = self.pz_env.agent_selection
    # print(f"agent now after step is {agent}")
    self._loop_others()
    observation, reward, terminated, truncated, info = (
      self.pz_env.observe(agent),
      self.pz_env.rewards[agent],
      self.pz_env.terminations[agent],
      self.pz_env.truncations[agent],
      self.pz_env.infos[agent]
    )
    # print(reward)
    return observation, reward, terminated, truncated, info

  def _loop_others(self):
    agent = self.pz_env.agent_selection
    # print(f"agent now is {agent}")
    while agent in self.act_others:
      act_current: ActOther = self.act_others[agent]
      obs_current: Observation = self.pz_env.observe(agent)
      action_current: Action = act_current(obs_current)
      self.pz_env.step(action_current)
      agent = self.pz_env.agent_selection

  def render(self, *args, **kwargs):
    self.pz_env.render(*args, **kwargs)
