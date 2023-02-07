import gym
from typing import Tuple, Callable
import numpy as np


class UpDownLeftRight(gym.ActionWrapper):
  """
  A gym environment wrapper to enable U/D/L/R action space when the wrapped environment
  actually expects a destination in cartesian coordinates.
  """

  def __init__(self, env: gym.Env,
               get_current_location: Callable[[], Tuple[int, int]]):

    super().__init__(env)
    self.get_current_location = get_current_location

    self.action_space = gym.spaces.Discrete(4)

    self.udlr_action_to_board_action = {
      0: (0, -1), # up
      1: (0, +1), # down
      2: (-1, 0), # left
      3: (+1, 0), # right
    }

  def action(self, action: int) -> Tuple[int, int]:
    current_location = self.get_current_location()
    return tuple(np.add(current_location, self.udlr_action_to_board_action[action]))
