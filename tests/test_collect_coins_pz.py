# import pytest

# import gym
from qwertyenv.ensure_valid_action_pz import EnsureValidAction

from tianshou.data import Collector
from tianshou.env import DummyVectorEnv, PettingZooEnv
from tianshou.policy import MultiAgentPolicyManager, RandomPolicy


from qwertyenv.collect_coins_pz import CollectCoinsEnv

from pettingzoo.test import api_test
# from pettingzoo.butterfly import pistonball_v6
# env = pistonball_v6.env()

# import pysnooper


# @pysnooper.snoop()
def test_evaluate():

  env = CollectCoinsEnv(pieces=['rock', 'rock'])

  action = None

  def another_action_taken(action_taken):
    nonlocal action
    action = action_taken

  # Wrapping the original environment as to make sure a valid action will be taken.
  env = EnsureValidAction(
      env,
      env.check_action_valid,
      env.provide_alternative_valid_action,
      another_action_taken
  )

  api_test(env, num_cycles=10, verbose_progress=False)


def test_with_tianshou():

  action = None

  # env =  gym.make('qwertyenv/CollectCoins-v0', pieces=['rock', 'rock'])

  env = CollectCoinsEnv(pieces=['rock', 'rock'], with_mask=True)

  def another_action_taken(action_taken):
    nonlocal action
    action = action_taken

  # Wrapping the original environment as to make sure a valid action will be taken.
  env = EnsureValidAction(
      env,
      env.check_action_valid,
      env.provide_alternative_valid_action,
      another_action_taken
  )

  env = PettingZooEnv(env)

  policies = MultiAgentPolicyManager([RandomPolicy(), RandomPolicy()], env)

  env = DummyVectorEnv([lambda: env])

  collector = Collector(policies, env)

  result = collector.collect(n_step=200, render=0.1)
