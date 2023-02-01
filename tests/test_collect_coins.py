import pytest

import gym
from qwertyenv import EnsureValidMove
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy


def test_evaluate():
  action = None

  env =  gym.make('qwertyenv/CollectCoins-v0', ['rock', 'rock'])

  def another_action_taken(action_taken):
    nonlocal action
    action = action_taken

  # Wrapping the original environment as to make sure a valid action will be taken.
  env = EnsureValidMove(
      env,
      env.check_action_valid,
      env.provide_alternative_valid_action,
      another_action_taken
  )
  agent_w = DQN(MlpPolicy, env)
  for episode in range(5):
    print(f'{episode=}')
    print("-------")
    obs = env.reset()
    env.render()
    while True:
      action, _state = agent_w.predict(obs)
      print(f'action predicted: {action}')
      obs, reward, terminated, truncated, info = env.step(action)
      done = terminated or truncated
      print(f'action taken: {action}')
      env.render()
      print(f'{reward=}')
      print(f'{done=}')
      print(f'{info=}')
      if done:
        break
    print()
