from pettingzoo.utils.wrappers import BaseWrapper
import gymnasium as gym


class ActionWrapper(BaseWrapper):
  def __init__(self, env: gym.Env):
    super().__init__(env)

  # def reset(self, seed=None, return_info=True, options=None):
  #   self.env.reset(seed=seed, return_info=return_info, options=options)

  def step(self, action):
    action = self.action(action)
    self.env.step(action)

  def action(self, action):
    pass

  def render(self, *args, **kwargs):
    self.env.render(*args, **kwargs)

  @property
  def agent_selection(self):
    return self.env.agent_selection
  
  @agent_selection.setter
  def agent_selection(self, agent):
    self.env.agent_selection = agent