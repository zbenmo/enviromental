from itertools import product
import gym
import numpy as np
import random
from typing import Protocol, List
from abc import ABC, abstractmethod


class Game(Protocol):
  ...


class Piece(ABC):
  """
  Either a rock or a knight (Chess like).
  """
  def __init__(self, game: Game, player: int) -> None:
    self.game = game
    self.player = player

  @abstractmethod
  def __repr__(self) -> str:
    pass
  
  @abstractmethod
  def valid_move(self, move) -> bool:
    pass


class Rock(Piece):
  """
  Rock
  """
  def __init__(self, game: Game, player: int) -> None:
    super().__init__(game, player)

  def __repr__(self) -> str:
    return f'{"w" if self.player == 0 else "b"}R'
  
  def valid_move(self, move) -> bool:
    current_location = self.game.locations[self.player]
    abs_move_row = abs(move[0] - current_location[0]) 
    abs_move_col = abs(move[1] - current_location[1]) 
    return (
      ((abs_move_row == 1) and (abs_move_col == 0))
      or
      ((abs_move_row == 0) and (abs_move_col == 1))
    )


class Knight(Piece):
  """
  Knight
  """
  def __init__(self, game: Game, player: int) -> None:
    super().__init__(game, player)

  def __repr__(self) -> str:
    return f'{"w" if self.player == 0 else "b"}K'
  
  def valid_move(self, move) -> bool:
    current_location = self.game.locations[self.player]
    abs_move_row = abs(move[0] - current_location[0]) 
    abs_move_col = abs(move[1] - current_location[1]) 
    return (
      ((abs_move_row == 1) and (abs_move_col == 2))
      or
      ((abs_move_row == 2) and (abs_move_col == 1))
    )


class CollectCoinsGame:
  """
  CollectCoinsGame
  """

  def __init__(self, pieces=['rock', 'rock']):
    assert len(pieces) == 2
    assert all(piece in ['rock', 'knight'] for piece in pieces)
    game = self
    self.pieces: List[Piece] = [
      Rock(game, i) if piece == 'rock' else Knight(game, i) for i, piece in enumerate(pieces)
    ]
    self.board = np.full((8, 8), True, dtype=bool)
    self.locations = [(0, 0), (7, 7)]
    for loc in self.locations:
      self.board[loc] = False
    self.coins = [0, 0]
    self.turn = 0

  def make_move(self, player, move) -> None:
    move = tuple(move)
    assert self.turn == player
    if move is not None:
      assert self.valid_move(player, move)
      self.coins[player] += int(self.board[move])
      self.board[move] = False
      self.locations[player] = move
    self.turn = 1 - self.turn

  def valid_move(self, player, move):
    move = tuple(move)
    if any(c < 0 or c > 7 for c in move):
      return False
    if any(location == move for location in self.locations):
      return False
    return self.pieces[player].valid_move(move)

  def render(self):
    horizontal = '-' * (self.board.shape[1] * 4 + 1)
    print(horizontal)
    for which_row, row in enumerate(self.board):
      row_copy = [f'{" "}{x}' for x in map({0: ' ', 1: '$'}.get, row)]
      for i, (player_location, player_piece) in enumerate(zip(self.locations, self.pieces)):
        if player_location[0] == which_row:
          row_copy[player_location[1]] = str(player_piece)
      print("|" + " |".join(row_copy) + " |")
      print(horizontal)
    print()
    print(f'{self.coins[0]}/{self.coins[1]}')

  def is_done(self) -> bool:
    return np.sum(self.board) < 1


class CollectCoinsEnv(gym.Env):
  """
  CollectCoins Gym environment.

  This a Chess like, two players, turns, board game.
  The aim of the game is to collect more coins than the opponent.
  On each turn a the playe's piece makes a move.
  If a coin is present in the destination the count is increased for the player.
  The game ends when there are not more coins on the board to collect. We then compare which player got the most.
  """

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

  def _calc_reward(self):
    current_coins = self.game.coins[self.player]
    previous_coins = self.previous_coins
    self.previous_coins = current_coins
    done = self.game.is_done()
    if done:
      draw = self.game.coins[self.player] == self.game.coins[self.other_player]
      win = self.game.coins[self.player] > self.game.coins[self.other_player]
      return 0 if draw else (1 if win else -1)
    else:
      return (current_coins - previous_coins) * 0.01

  def render(self, *args, **argv):
    self.game.render()

  def check_action_valid(self, action, player=None) -> bool:
    """
    This helper function can be used when initializing EnsureValidAction wrapper as an example (see in examples).
    """

    return self.game.valid_move(
      (self.player if player is None else player),
      action
    )

  def provide_alternative_valid_action(self, action, player=None):
    """
    This helper function can be used when initializing EnsureValidAction wrapper as an example (see in examples).

    Select a random valid move.
    TOOD: maybe consult the provided action(move) and provide an action that is as close as possible.
    """

    all_moves = product(range(8), repeat=2)
    all_valid_moves = [move for move in all_moves if self.check_action_valid(move, player)]
    return None if len(all_valid_moves) < 1 else random.choice(all_valid_moves)
