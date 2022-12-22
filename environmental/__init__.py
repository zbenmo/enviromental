from gym.envs.registration import register

__version__ = "0.0.1"


register(
    id='environmental/BlackJack-v0',
    entry_point='environmental.black_jack:BJEnv',
    max_episode_steps=300,
)