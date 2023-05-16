# BJEnv - BlackJack

This casino-style game is described in RLBook2018.
The player needs to decide wheater to ask for an additional card (hits), or to pass the turn to the dealer (sticks).
As long as the player takes additional cards and was not busted (sum of cards is still <= 21), the turn stays with the player. 
Then the dealer plays (with a fixed rule when to take additional card and when to stop).
Finally the player wins when they were not busted, and have a sum that is bigger then the dealer's sum (or the player were not busted yet the dealer was).

An Ace can be counted as 1 or as 11. Therefore the first Ace is "useful".
A second Ace shall always be counted as 1 to avoid > 21 sum (given that the first Ace was counted as 11).

The implementation found here in qwertyenv is a Gymnasium environment for the player (you should build a policy for the player).

In RLBook2018 this environment is brought in the context of MC with "Exploring Starts".
Cards are selected not from a deck but rather with randomness from all possibilities (so with repetitions). To the best of my memory this is what is described in the RLBook2018.

Below you can see are the action_space and the observation_space.

Please note that experimenting MC "Exploring Starts" is facilitates as the environment itself contains the uniform randomness for the starting state. We start the game when the player has already at least a sum of 11, as otherwise the only "smart" action is 'hits'.

``` py title="part of BJEnv Gymnasium environment"
self.action_space = gym.spaces.Discrete(2) # 0 hits, 1 sticks

obs_space = dict(
    player_sum = gym.spaces.Discrete(11), # x -> x + 11
    player_useful_Ace = gym.spaces.Discrete(2), # 0 no, 1 yes
    dealer_shown_card = gym.spaces.Discrete(10) # 0 - Ace, x -> x + 1 (ex. 1 is 2, 9 is 10)
)

self.observation_space = gym.spaces.Dict(obs_space)
```
