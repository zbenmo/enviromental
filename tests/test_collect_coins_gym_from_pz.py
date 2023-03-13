from qwertyenv.ensure_valid_action_pz import EnsureValidAction
from qwertyenv.pz_to_gymnasium_wrapper import PZ2GymnasiumWrapper

from tianshou.data import Collector
from tianshou.env import DummyVectorEnv
from tianshou.policy import RandomPolicy


from qwertyenv.collect_coins_pz import CollectCoinsEnv

from gymnasium.utils.env_checker import check_env


def test_evaluate():

  env = CollectCoinsEnv(pieces=['rock', 'rock']) # Note: PettingZoo environment (AECenv)

  action = None

  def another_action_taken(action_taken):
    nonlocal action
    action = action_taken

  # Wrapping the original environment as to make sure a valid action will be taken.
  env = EnsureValidAction(  # Still a PettingZoo environment
      env,
      env.check_action_valid,
      env.provide_alternative_valid_action,
      another_action_taken
  )

  def act_agent_1(obs):
      return (0, 0)

  env = PZ2GymnasiumWrapper(env, act_others={'agent_1': act_agent_1}) # Note: Gymnasium environment

  check_env(env) # , num_cycles=10, verbose_progress=False)


def test_with_tianshou():

  action = None

  # env =  gym.make('qwertyenv/CollectCoins-v0', pieces=['rock', 'rock'])

  env = CollectCoinsEnv(pieces=['rock', 'rock'], with_mask=True)  # Note: PettingZoo environment (AECenv)

  def another_action_taken(action_taken):
    nonlocal action
    action = action_taken

  # Wrapping the original environment as to make sure a valid action will be taken.
  env = EnsureValidAction(  # Still a PettingZoo environment
      env,
      env.check_action_valid,
      env.provide_alternative_valid_action,
      another_action_taken
  )

  def act_agent_1(obs):
      return (0, 0)

  env = PZ2GymnasiumWrapper(env, act_others={'agent_1': act_agent_1}) # Note: Gymnasium environment

  env = DummyVectorEnv([lambda: env])

  policy = RandomPolicy()

  collector = Collector(policy, env)

  result = collector.collect(n_step=200, render=0.1)
