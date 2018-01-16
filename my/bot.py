import numpy as np
import math, random
import hlt
from my.estimator import Estimator, fight
from my.clustering import all_clusters
from my.features import my_ships_features


class Bot:
  """Responsible for playing the game."""
  
  def __init__(self, estimator, name):
    """Load estimators, in this case, they are all linear regressors. And
    set the in-game name of the bot."""
    self.estimator = estimator
    self._name = name
  
  def play(self):
    """Play a game using stdin/stdout."""
    game = hlt.Game(self._name)
    
    while True:
      game_map = game.update_map()
      clusters = all_clusters(game_map)
      s_feats = my_ships_features(game_map, clusters)
      command_queue = []
      
      # Determine the course of action for each ship independently.
      for feats, ship in zip(s_feats, game_map.get_me().all_ships()):
        
        # Fighter line of decision.
        if ship.docking_status == ship.DockingStatus.UNDOCKED:
          
          # Find planets where we can dock.
          dock_targets = []
          for p in game_map.all_planets():
            if p.is_full() or not ship.can_dock(p):
              continue
            if p.is_owned() and ship.owner is not p.owner:
              continue
            dock_targets.append(p)
          
          # If possible, dock to a randomly chosen planet from those where we can dock.
          if len(dock_targets) > 0:
            target = random.choice(dock_targets)
            command_queue.append(ship.dock(target))
            continue
          
          # Otherwise, move.
          speed, angle = fight(feats, self.estimator)
          command_queue.append(ship.thrust(speed, angle))
        
        # Miner line of decision.
        # Do nothing (continue mining).
      
      game.send_command_queue(command_queue)
