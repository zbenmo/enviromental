from gym.envs.registration import register
from .ensure_valid_action import EnsureValidAction
from .up_down_left_right import UpDownLeftRight

__version__ = "0.0.3"


register(
    id='qwertyenv/BlackJack-v0',
    entry_point='qwertyenv.black_jack:BJEnv',
    max_episode_steps=300
)

register(
    id='qwertyenv/CollectCoins-v0',
    entry_point='qwertyenv.collect_coins:CollectCoinsEnv',
    max_episode_steps=300
)

__all__ = [
    EnsureValidAction,
    UpDownLeftRight
]