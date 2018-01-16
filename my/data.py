import math, random
import numpy as np
from hlt.entity import Planet, Ship
from hlt.game_map import Map, Player
from my.clustering import all_clusters
from my.features import ship_features, FEATURES, INDICATORS
from my.estimator import identity


def get_maps(data):
  """Constructs a list of hlt.Map objects (corresponding to
  the states of the game in frames 1, 2, ...) from <data>
  (which is in replay format)."""
  res = []
  
  num_players = data["num_players"]
  
  num_frames = data["num_frames"]
  width = data["width"]
  height = data["height"]
  
  for fid in range(num_frames):
    game_map = Map(None, width, height)
    frame = data["frames"][fid]
    tokens = []
    
    # Generate tokens for players (their ships).
    tokens.append(num_players)
    for pid, ships in frame["ships"].items():
      tokens.append(pid)
      tokens.append(len(ships))
      for sid, ship in ships.items():
        docked = ship["docking"]["status"]
        docked = {"undocked": 0, "docking": 1, "docked": 2, "undocking": 3}[docked]
        planet = ship["docking"].get("planet_id", -1)
        progress = ship["docking"].get("turns_left", -1)
        cooldown = ship["cooldown"]
        
        tokens.extend(map(lambda key: ship[key], ["id", "x", "y", "health", "vel_x", "vel_y"]))
        tokens.extend([docked, planet, progress, cooldown])
    
    # Generate tokens for planets.
    tokens.append(len(frame["planets"]))
    for curr in frame["planets"].values():
      plid = curr["id"]
      origin = data["planets"][plid]
      x = origin["x"]
      y = origin["y"]
      hp = curr["health"]
      r = origin["r"]
      docking = origin["docking_spots"]
      current = curr["current_production"]
      remaining = curr["remaining_production"]
      owner = curr["owner"]
      owned = (1 if owner is not None else 0)
      if not owned:
        owner = 0
      docked_ships = curr["docked_ships"]
      num_docked_ships = len(docked_ships)
      
      tokens.extend([plid, x, y, hp, r, docking, current, remaining, owned, owner, num_docked_ships])
      tokens.extend(docked_ships)
    
    # Create a string representation, and let it be parsed to create a Map.
    tokens = map(str, tokens)
    game_map._parse(' '.join(tokens))
    res.append(game_map)
  
  return res

#########################################################################
#### DICT CONVENIENCE METHODS ###########################################

def set_val(location, value, res):
  """A helper method to conveniently set values in nested dictionaries."""
  curr = res
  for name in location[:-1]:
    if name not in curr:
      curr[name] = {}
    curr = curr[name]
  last_name = location[-1]
  curr[last_name] = value

def get_val(location, res, default = None):
  """A helper method to conveniently get values in nested dictionaries."""
  curr = res
  for name in location:
    if name not in curr:
      return default
    curr = curr[name]
  return curr

def add_val(location, value, res):
  """A helper method to conveniently modify numeric values in nested dictionaries."""
  prev_val = get_val(location, res, default = 0.0)
  set_val(location, prev_val + value, res)

def attack(fid, sid, targets, res):
  """Records the event in <res>. The event took place in frame <fid>,
  ship <sid> attacked all ships in <targets> and spread its 64 damage
  evenly."""
  damage = 64 / len(targets)
  for tgt in targets:
    add_val([sid, fid, "attack", tgt], damage, res)
    add_val([tgt, fid, "attacked_by", sid], damage, res)

def spawn(fid, sid, planet_id, res):
  """Records the event 'ship with id <sid> has spawned at frame <fid>'."""
  set_val([sid, "spawned"], (fid, planet_id), res)

def destroy(fid, sid, res):
  """Records the event 'ship with id <sid> was destroyed at frame <fid>'."""
  set_val([sid, "destroyed"], fid, res)

#########################################################################
#### EVENTS AND MOVES ###################################################

def get_moves(data):
  """Returns a dictionary that contains for each ship and each frame
  the move taken by that ship in that frame, if any."""
  res = {}
  for fid, frame in enumerate(data["moves"][:-1]):
    for pid, content in frame.items():
      for move in content[0].values():
        sid = move["shipId"]
        t = move["type"]
        if t == "thrust":
          phi = math.radians(move["angle"])
          speed = move["magnitude"]
          dx = speed * math.cos(phi)
          dy = speed * math.sin(phi)
          set_val([sid, fid], (t, dx, dy), res)
        else:
          set_val([sid, fid], (t,), res)
  return res

def get_events(data):
  """For each ship and each frame, create a list of events containing that
  ship, except for 'spawn' and 'destroyed'. For each ship, remember
  the frame number when it spawned and when it died."""
  res = {}
  for fid, frame in enumerate(data["frames"][:-1]):
    for ev in frame["events"]:
      sid = ev["entity"]["id"]
      if ev["event"] == "attack":
        targets = list(map(lambda x: x["id"], ev["targets"]))
        attack(fid, sid, targets, res)
      elif ev["event"] == "spawned":
        spawn(fid, sid, ev["planet"]["id"], res)
      elif ev["event"] == "destroyed":
        destroy(fid, sid, res)
      # Ignore contention attacks for now.
  return res

#########################################################################
#### REWARDS ############################################################

def get_rewards(frame_maps, events):
  """
  For each ship and each frame (except for the last one), calculates
  the 'reward' received by that ship. The size of the reward is
  proportional to the difference in total health and ship numbers:
    +255 points for attacking a ship that was destroyed this turn
    +255 points whenever a friendly ship spawns on a planet where we are docked
    +1 point for each damage dealt
    -1 point for each damage taken
    -255 points for being destroyed
  """
  res = {}
  for sid, content in events.items():
    for a, b in content.items():
      if a == "spawned":
        fid, pid = b
        for ship in frame_maps[fid].get_planet(pid).all_docked_ships():
          add_val([ship.id, fid], 255, res)
      elif a == "destroyed":
        fid = b
        add_val([sid, fid], -255, res)
        
        # Give points to those who took this ship down.
        contributors = get_val([fid, "attacked_by"], content, default = {})
        for c in contributors:
          add_val([c, fid], 255, res)
      else:
        fid = a
        
        # Get points for each point of damage dealt.
        dealt_damage_to = get_val(["attack"], b, default = {})
        for dmg in dealt_damage_to.values():
          add_val([sid, fid], dmg, res)
        
        # Lose points for each point of damage taken.
        damage_taken_from = get_val(["attacked_by"], b, default = {})
        for dmg in damage_taken_from.values():
          add_val([sid, fid], -dmg, res)
  
  return res

def get_utilities(rewards, num_frames, discount, max_len):
  """From the received rewards, calculate the utilities (which take into
  account future rewards)."""
  res = {}
  for sid, content in rewards.items():
    spawn_fid = content.get("spawned", -1)
    curr = 0.0
    subres = {}
    for fid in range(num_frames - 1, spawn_fid, -1):
      curr *= discount
      curr += content.get(fid, 0.0)
      subres[fid] = curr
    for fid in range(num_frames - 1, spawn_fid, -1):
      subres[fid] -= subres.get(fid + max_len, 0.0) * discount**max_len
    res[sid] = subres
  return res

#########################################################################
#### DATA CREATION ######################################################

SHIP_DESCRIPTION = FEATURES + ["dx", "dy"] + INDICATORS + ["thrust", "dock", "undock"]

def feats_to_list(feats):
  """Returns the features as a list of their values in the order defined by
  SHIP_DESCRIPTION."""
  return list(map(lambda x: feats[x], SHIP_DESCRIPTION))

def feats_from_list(src):
  """Returns a dictionary of things that describe the ship given by the list <src>."""
  return {name: src[i] for i, name in enumerate(SHIP_DESCRIPTION)}

def to_table(data, sample_ratio = 0.1, discount = 0.95, max_len = 50, skip_tail = True, skip_short_game = True):
  """Returns a numpy array where all columns except for the last are
  the (original) attributes, and the last column is the attribute to be
  predicted: the utility."""
  
  max_frame = data["num_frames"] - (max_len if skip_tail else 1)
  if skip_short_game and max_frame <= 2 * max_len:
    print("Game too short, skipping...")
    return np.zeros((0, len(SHIP_DESCRIPTION) + 1))
  
  res = []
  
  frame_maps = get_maps(data)
  events = get_events(data)
  moves = get_moves(data)
  rewards = get_rewards(frame_maps, events)
  utilities = get_utilities(rewards, len(frame_maps), discount, max_len)
  
  for fid, game_map in enumerate(frame_maps[: max_frame]):
    if random.random() >= sample_ratio:
      continue
    print("Frame {}".format(fid + 1))
    
    clusters = all_clusters(game_map)
    planets = game_map.all_planets()
    
    for ship in game_map._all_ships():
      if ship.docking_status != ship.DockingStatus.UNDOCKED:
        continue
      sid = ship.id
      move = get_val([sid, fid], moves, default = ("thrust", 0.0, 0.0))
      
      # Get the description of the ship.
      feats = ship_features(ship, clusters, planets)
      feats["dock"] = (1 if move[0] == "dock" else 0)
      feats["undock"] = (1 if move[0] == "undock" else 0)
      feats["thrust"] = (1 if move[0] == "thrust" and not feats["docked"] else 0)
      if feats["thrust"]:
        feats["dx"] = move[1]
        feats["dy"] = move[2]
      else:
        feats["dx"] = 0.0
        feats["dy"] = 0.0
      
      u = get_val([sid, fid], utilities, default = 0.0)
      subres = feats_to_list(feats)
      subres.append(u)
      res.append(subres)
  
  print("Framing done.")
  return np.array(res)

def get_Xy(table, expander = identity):
  """Selects the appropriate rows from <table>."""
  X = []
  y = []
  for row in table:
    feats = feats_from_list(row)
    if feats["thrust"] != 1:
      continue
    X.append(expander(feats))
    y.append(row[-1])
  X = np.array(X)
  y = np.array(y)
  return X, y
