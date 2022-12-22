import gym
import random


class BJEnv(gym.Env):
    """
    Blackjack
    """
    
    class State:
        def __init__(self):
            # for the player I start with the sum rather than the cards to assure uniformity in the randomized start
            self.player_sum = random.randint(11, 21)
            # and so for the useful Ace
            self.player_useful_Ace = random.randint(0, 1) == 1 # boolean

            self.dealer_cards = [self._random_card(), self._random_card()]

        def _random_card(self):
            return random.randint(1, 10) # 1 - Ace, 2, .. 10

        def hits(self):
            new_card = self._random_card()
            self.player_sum += new_card # note since we start with 11 we cannot add another ACE.
            if self.player_sum > 21:
                if self.player_useful_Ace:
                    self.player_sum -= 10
                    self.player_useful_Ace = False
            return new_card
        
        def stick(self):                    
            return self._dealers_turn()
            
        def _dealers_turn(self):
            cur_sum = sum(self.dealer_cards)
            if cur_sum <= 11 and 1 in self.dealer_cards:
                cur_sum += 10
            while cur_sum < 17:
                self.dealer_cards.append(self._random_card())
                cur_sum = sum(self.dealer_cards)
                if cur_sum <= 11 and 1 in self.dealer_cards:
                    cur_sum += 10
            return cur_sum

        def render(self):
            if self.player_useful_Ace:
                print("Useful Ace")
            else:
                print("No useful Ace")
            print("Current sum: ", self.player_sum)
            print("Dealer showing: ",  self.card_to_image(self.dealer_cards[0]))
            
        def card_to_image(self, value):
            return "A" if value == 1 else value

                    
    def __init__(self):
        self.action_space = gym.spaces.Discrete(2) # 0 hits, 1 sticks
        
        obs_space = dict(
            player_sum = gym.spaces.Discrete(11), # x -> x + 11
            player_useful_Ace = gym.spaces.Discrete(2), # 0 no, 1 yes
            dealer_shown_card = gym.spaces.Discrete(10) # 0 - Ace, x -> x + 1 (ex. 1 is 2, 9 is 10)
        )
        
        self.observation_space = gym.spaces.Dict(obs_space)
        self.reset()
    def render(self):
        self.state.render()
    def reset(self, seed=None, options=None):
        if seed:
          random.seed(seed)
        self.state = BJEnv.State()
        return self._state_to_obs(), {}
    def _state_to_obs(self):
        return {
            'player_sum' : self.state.player_sum - 11,
            'player_useful_Ace': 1 if self.state.player_useful_Ace else 0,
            'dealer_shown_card' : self.state.dealer_cards[0] - 1
        }
    def step(self, action):
        done = None
        reward = None
        info = {}
        if action == 1: # sticks
            dealer_sum = self.state.stick()
            if dealer_sum > 21:
                reward = 1 # dealer busted
            elif self.state.player_sum < dealer_sum:
                reward = -1 # lose
            elif self.state.player_sum == dealer_sum:
                reward = 0 # draw
            else:
                reward = 1 # win
            info['dealer sum'] = dealer_sum 
            done = True
        elif action == 0: # hits
            new_card = self.state.hits()
            done = False
            reward = 0
            info['new card'] = self.state.card_to_image(new_card)
            if self.state.player_sum > 21:
                reward = -1
                done = True
        else:
            assert False, f"unkown action {action}"
        terminated, truncated = done, False
        return self._state_to_obs(), reward, terminated, truncated, info