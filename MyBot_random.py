import random
import hlt


game = hlt.Game("Random")

while True:
  game_map = game.update_map()
  
  command_queue = []
      
  for ship in game_map.get_me().all_ships():
    
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
      
      # Otherwise, move randomly.
      speed = random.randint(random.randint(0, 7), 7)
      angle = random.randint(0, 359)
      command_queue.append(ship.thrust(speed, angle))
    
    # Miner line of decision.
    # Do nothing (continue mining).
  
  game.send_command_queue(command_queue)
