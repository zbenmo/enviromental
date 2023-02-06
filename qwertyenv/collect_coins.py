from itertools import product
import gym
import numpy as np
import random


class CollectCoinsGame:
  def __init__(self, pieces=['rock', 'rock']):
    assert len(pieces) == 2
    assert all(piece in ['rock', 'knight'] for piece in pieces)
    self.pieces = pieces
    self.board = np.full((8, 8), True, dtype=bool)
    self.locations = [(0, 0), (7, 7)]
    for loc in self.locations:
      self.board[loc] = False
    self.coins = [0, 0]
    self.turn = 0

  def make_move(self, player, move) -> None:
    move = tuple(move)
    # assert isinstance(move, tuple)
    # assert move.shape == 2
    # assert 0 <= move[0] < 8
    # assert 0 <= move[1] < 8
    assert self.turn == player
    if move is not None:
      # assert self.board[tuple(move)] in [' ', self.symbols[player]], f'{move} is actually: {self.board[tuple(move)]}'
      assert self.valid_move(player, move)
      self.coins[player] += int(self.board[move])
      self.board[move] = False
      self.locations[player] = move
    self.turn = 1 - self.turn

  def valid_move(self, player, move):
    move = tuple(move)
    if any(location == move for location in self.locations):
      return False
    # if self.board[tuple(move)] not in [' ', self.symbols[player]]:
    #   return False
    current_location = self.locations[player]
    if self.pieces[player] == 'knight':
      abs_move_row = abs(move[0] - current_location[0]) 
      abs_move_col = abs(move[1] - current_location[1]) 
      return (
        ((abs_move_row == 1) and (abs_move_col == 2))
        or
        ((abs_move_row == 2) and (abs_move_col == 1))
      )
    elif self.pieces[player] == 'rock':
      abs_move_row = abs(move[0] - current_location[0]) 
      abs_move_col = abs(move[1] - current_location[1]) 
      return (
        ((abs_move_row == 1) and (abs_move_col == 0))
        or
        ((abs_move_row == 0) and (abs_move_col == 1))
      )

  def render(self):
    horizontal = '-' * (self.board.shape[1] * 4 + 1)
    print(horizontal)
    for which_row, row in enumerate(self.board):
      row_copy = [f'{" "}{x}' for x in map({0: ' ', 1: '$'}.get, row)]
      for i, (player_location, player_piece) in enumerate(zip(self.locations, self.pieces)):
        if player_location[0] == which_row:
          row_copy[player_location[1]] = f'{"w" if i == 0 else "b"}{"R" if player_piece == "rock" else "K"}' 
      print("|" + " |".join(row_copy) + " |")
      print(horizontal)
    print()
    print(f'{self.coins[0]}/{self.coins[1]}')

  def is_done(self) -> bool:
    return np.sum(self.board) < 1


class CollectCoinsEnv(gym.Env):
  def __init__(self, pieces=['rock', 'rock'], player=0):
    """
    player 0 is the first entry, AKA white
    player 1 is the second entry, AKA black
    """
    obs_space = dict(
      board = gym.spaces.Box(low=0, high=1, shape=(8 * 8,), dtype=bool),
      player = gym.spaces.MultiDiscrete([8, 8]),
      other_player = gym.spaces.MultiDiscrete([8, 8]),
    )    
    self.observation_space = gym.spaces.Dict(spaces=obs_space)
    # self.observation_space = gym.spaces.Box(low=0, high=1, shape=(8 * 8,), dtype=bool)
    self.action_space = gym.spaces.MultiDiscrete([8, 8])

    self.pieces = pieces
    self.player = player
    self.other_player = 1 - player
    self.game = None
    self.previous_coins = None
    self.reset()

  def reset(self, seed=None, options=None):
    self.game = CollectCoinsGame(self.pieces)
    self.previous_coins = 0
    return self._get_observation() 

  def step(self, action):
    self.game.make_move(self.player, action)
    if not self.game.is_done():
      self._play_other()
    info = {}
    return self._get_observation(), self._calc_reward(), self.game.is_done(), info

  def _play_other(self):
    """
    a random move for now
    """
    move = self.provide_alternative_valid_action(None, self.other_player)
    self.game.make_move(self.other_player, move)

  def _get_observation(self):
    return dict(
      board=self.game.board.flatten(),
      player=self.game.locations[self.player],
      other_player=self.game.locations[self.other_player]
    )
    # return self.game.board.flatten()

  def _calc_reward(self):
    current_coins = self.game.coins[self.player]
    previous_coins = self.previous_coins
    self.previous_coins = current_coins
    done = self.game.is_done()
    if done:
      draw = self.game.coins[self.player] == self.game.coins[self.other_player]
      win = self.game.coins[self.player] > self.game.coins[self.other_player]
      return 0 if draw else (100 if win else -100)
    else:
      return current_coins - previous_coins

  def render(self, *args, **argv):
    self.game.render()

  def check_action_valid(self, action, player=None) -> bool:
    """
    This helper function can be used when initializing EnsureValidMove wrapper as an example (see in examples).
    """
    return self.game.valid_move(
      (self.player if player is None else player),
      action
    )

  def provide_alternative_valid_action(self, action, player=None):
    """
    This helper function can be used when initializing EnsureValidMove wrapper as an example (see in examples).

    Select a random valid move.
    TOOD: maybe consult the provided action(move) and provide an action that is as close as possible.
    """
    all_moves = product(range(8), repeat=2)
    all_valid_moves = [move for move in all_moves if self.check_action_valid(move, player)]
    # print(f'#moves = {len(all_valid_moves)}')
    # print(f'{all_valid_moves=}')
    return None if len(all_valid_moves) < 1 else random.choice(all_valid_moves)
