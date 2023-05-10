import pytest

import gymnasium as gym
from qwertyenv import EnsureValidAction
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MultiInputPolicy
from stable_baselines3.common.evaluation import evaluate_policy


def test_evaluate():
  action = None

  env =  gym.make('qwertyenv/CollectCoins-v0', pieces=['rock', 'rock'])

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
  agent_w = PPO(MultiInputPolicy, env)
  for episode in range(1):
    # print(f'{episode=}')
    # print("-------")
    obs, _ = env.reset()
    # env.render()
    while True:
      action, _state = agent_w.predict(obs)
      # print(f'action predicted: {action}')
      obs, reward, terminated, truncated, info = env.step(action)
      done = terminated or truncated
      # print(f'action taken: {action}')
      # env.render()
      # print(f'{reward=}')
      # print(f'{done=}')
      # print(f'{info=}')
      if done:
        break
    # print()

  mean_reward, std_reward = evaluate_policy(agent_w, env, deterministic=True)
