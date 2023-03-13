from pettingzoo.utils.wrappers import BaseWrapper
import gymnasium as gym


class ActionWrapper(BaseWrapper):
  def __init__(self, env: gym.Env):
    super().__init__(env)

  def step(self, action):
    action = self.action(action)
    self.env.step(action)

  def action(self, action):
    pass

  def render(self, *args, **kwargs):
    self.env.render(*args, **kwargs)
