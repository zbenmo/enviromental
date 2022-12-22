from gym.envs.registration import register

__version__ = "0.0.1"


register(
    id='qwertyenv/BlackJack-v0',
    entry_point='qwertyenv.black_jack:BJEnv',
    max_episode_steps=300,
)