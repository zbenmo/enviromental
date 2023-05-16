# Wrapping a PettingZoo into a Gymnasium one

Two utility functions 'aec_to_gymnasium' and the 'parallel_to_gymnasium' are provided in qwertyenv.

In order to have a PettingZoo environment presenting itself as a Gymnasium environment, we need to know which of the agents is the "external agent", the one that should not be "wrapped into" the Gymnasium environment.
In addition we need to know how to get the actions for all other agents.

``` py
from typing import Any, Callable, Dict, Optional

import gymnasium as gym
from gymnasium.core import ActType, ObsType

from pettingzoo import AECEnv, ParallelEnv


# The first parameter is an agent (its identification),
#  second parameter is the relevant observation.
# The callable is expected to return the action that the
#  given agent would like to take.
ActOthers = Callable[[str, ObsType], ActType]


def aec_to_gymnasium(aec_env: AECEnv,
  external_agent: str, act_others: ActOthers):
    """Makes a Gymnasium environment out of a AECEnv.
    ...
    """
...


def parallel_to_gymnasium(parallel_env: ParallelEnv,
  external_agent: str, act_others: ActOthers):
    """Makes a Gymnasium environment out of a ParallelEnv.
    ...
    """
...
```

Below is a brief description of the implementation.
For both functions, I return an instance of a (inner) class defined in the function itself ```class WrapperEnv(gym.Env)```. 
'aec_to_gymnasium' was a bit trickier. I had to check if the current agent is the 'external_agent' or is it another agent. I used the following member function of my 'WrapperEnv' class.

``` py title='part of (inner) class WrapperEnv(gym.Env) - aec_to_gymnasium'
def _loop_others(self):
    for agent in self._aec_env.agent_iter():
        if agent == self._external_agent:
            break
        observation, _, terminated, truncated, _ = self._aec_env.last()
        if terminated or truncated:
            break
        action_current = self._act_others(agent, observation)
        self._aec_env.step(action_current)
```

For 'parallel_to_gymnasium' it was just a question of where to take the next action, from the provided argument to the 'step' method, or from 'act_others' callable that was provieded in the initialization.

``` py title='part of (inner) class WrapperEnv(gym.Env) - parallel_to_gymnasium'
def step(self, action):
  ...
  actions = {
      agent: (
          action
          if agent == self._external_agent
          else self._act_others(agent, self._observations[agent])
      )
      for agent in self._parallel_env.agents
  }
  ...
```

An example for the usage of those functions, is as follows:

``` py
def test_tictactoe(me, other):
    aec_env = tictactoe_v3.env()

    def pick_a_free_square(obs):
        action_mask = obs["action_mask"]
        possible_actions = np.where(action_mask == 1)[0]
        return np.random.choice(possible_actions)

    other_agents_logic = {other: pick_a_free_square}

    gym_env = aec_to_gymnasium(
        aec_env=aec_env,
        external_agent=me,
        act_others=(
          lambda agent, observation: other_agents_logic[agent](observation)
        ),
    )
...
```

In above example, the lambda function 'act_others' is making a use of a dict 'other_agents_logic'. In this simple case, we could also just ignore the 'agent' argument, and call 'pick_a_free_square' directly.
