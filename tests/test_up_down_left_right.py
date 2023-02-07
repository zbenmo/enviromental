import pytest

import gym
from qwertyenv import UpDownLeftRight


def test_up_down_left_right():
  env =  gym.make('qwertyenv/CollectCoins-v0', pieces=['rock', 'rock'])

  env = UpDownLeftRight(env, lambda: (5, 5))
  assert env.action(0) == (5, 4)
  assert env.action(1) == (5, 6)
  assert env.action(2) == (4, 5)
  assert env.action(3) == (6, 5)
