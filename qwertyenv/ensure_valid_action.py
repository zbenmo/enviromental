from typing import TypeVar, Callable
import gymnasium as gym


Action = TypeVar("Action")


class EnsureValidAction(gym.ActionWrapper):
  """
  A gym environment wrapper to help with the case that the agent wants to take invalid actions.
  For example consider a Chess game, where you let the action_space be any piece moving to any square on the board,
  but then when a wrong move is taken, instead of returing a big negative reward, you just take another action,
  this time a valid one. To make sure the learning algorithm is aware of the action taken, a callback should be provided.
  """
  def __init__(self, env: gym.Env,
    check_action_valid: Callable[[Action], bool],
    provide_alternative_valid_action: Callable[[Action], Action],
    alternative_action_cb: Callable[[Action], None]):

    super().__init__(env)
    self.check_action_valid = check_action_valid
    self.provide_alternative_valid_action = provide_alternative_valid_action
    self.alternative_action_cb = alternative_action_cb

  def action(self, action: Action) -> Action:
    if self.check_action_valid(action):
      return action
    alternative_action = self.provide_alternative_valid_action(action)
    self.alternative_action_cb(alternative_action)
    return alternative_action