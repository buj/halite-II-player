import numpy as np

import logging


# Various sensor functions. A sensor function is one that
# accepts two arguments: the source (who is sensing) and the
# target (who is being sensed), and returns a number that tells us
# how much does the sensor fire. (For example, the sensor can measure
# proximity to a planet.)

def distance(p, q):
  return ((p.x - q.x)**2 + (p.y - q.y)**2)**0.5

def proximity(p, q):
  if p is q:
    return 0.0
  return 1.0 / max(1.0, distance(p, q) - p.radius - q.radius)**2

def number(pos, cluster):
  if cluster.dist_min(pos) <= cluster.radius or len(cluster.ships) <= 3:
    return sum(map(lambda s: proximity(s, pos), cluster.ships))
  return cluster.size / cluster.dist_geo(pos)**2

def health(pos, cluster):
  if cluster.dist_min(pos) <= cluster.radius or len(cluster.ships) <= 3:
    return sum(map(lambda s: s.health * proximity(s, pos), cluster.ships))
  return cluster.health / cluster.dist_geo(pos)**2

def free_docks(ship, planet):
  if (not planet.is_owned()) or (planet.owner is ship.owner):
    remaining_docks = planet.num_docking_spots - len(planet.all_docked_ships())
    return remaining_docks * proximity(ship, planet)
  return 0.0

# Various projections. A projection is a function which takes as input
# the source (who is sensing), the target (who is being sensed), and the
# value of the sensor (how much does it fire), and returns a projection of
# the sensor's firing rate (for example, onto the x-axis, in the up direction, ...).

def id_(start, end, val):
  return val

def x_(start, end, val):
  dist = distance(start, end)
  ratio = ((end.x - start.x) / dist if dist > 0.0 else 1.0)
  return val * ratio

def y_(start, end, val):
  dist = distance(start, end)
  ratio = ((end.y - start.y) / dist if dist > 0.0 else 1.0)
  return val * ratio

def down_(start, end, val):
  return max(y_(start, end, val), 0.0)

def up_(start, end, val):
  return max(-y_(start, end, val), 0.0)

def right_(start, end, val):
  return max(x_(start, end, val), 0.0)

def left_(start, end, val):
  return max(-x_(start, end, val), 0.0)

"""Dict of directional projections."""
DIRECTIONS = ["up", "down", "right", "left"]
dir_projs = {"up": up_, "down": down_, "right": right_, "left": left_}


def fire(p, q, sensor_func, proj_func = id_):
  """How much does the sensor <val_func> fire when projected by <proj_func>?"""
  return proj_func(p, q, sensor_func(p, q))


#######################################################################
#### SHIP STUFF #######################################################

"""Dict of sensors."""
ship_sensors = {"size": number, "health": health}
planet_sensors = {"proximity": proximity, "docks": free_docks}

"""Basic ship features (not yet expanded)."""
INDICATORS = [
  "docked"
]
STAT_FEATURES = [
  "health"
]
DIR_FEATURES = []

for who in ["ally", "enemy"]:
  for ship_type in ["miners", "fighters"]:
    for sensor in ship_sensors.keys():
      for d in DIRECTIONS:
        DIR_FEATURES.append("{}_{}_{}_{}".format(who, ship_type, sensor, d))
for what in planet_sensors.keys():
  for d in DIRECTIONS:
    DIR_FEATURES.append("{}_{}".format(what, d))

FEATURES = STAT_FEATURES + DIR_FEATURES


def ship_features(ship, clusters, planets):
  """Calculate features for <ship>."""
  res = {
    "docked": (1 if ship.docking_status == ship.DockingStatus.DOCKED else 0),
    "health": ship.health / 255.0
  }
  
  # Number sense: how many ships are there in this direction?
  # Health sense: how healthy are ships in this direction?
  ship_data = []
  for player_clusters in clusters:
    by_ship_type = {}
    for ship_type, sub_clusters in player_clusters.items():
      by_sensor = {}
      for sensor, sensor_func in ship_sensors.items():
        by_direction = {}
        for proj, proj_func in dir_projs.items():
          by_direction[proj] = sum(map(lambda c: fire(ship, c, sensor_func, proj_func), sub_clusters))
        by_sensor[sensor] = by_direction
      by_ship_type[ship_type] = by_sensor
    ship_data.append(by_ship_type)
  
  # Put it into <res>.
  for who in ["ally", "enemy"]:
    for ship_type in ["miners", "fighters"]:
      for sensor in ship_sensors.keys():
        for direction in DIRECTIONS:
          key = "{}_{}_{}_{}".format(who, ship_type, sensor, direction)
          if who == "ally":
            res[key] = ship_data[ship.owner.id][ship_type][sensor][direction]
          else:
            res[key] = max([ship_data[i][ship_type][sensor][direction] for i in range(len(ship_data)) if i != ship.owner.id], default = 0.0)
  
  # Collision sense: don't crash into planet!
  # Objective sense: aim for nearby planets with many docking spots.
  for sensor, sensor_func in planet_sensors.items():
    for proj, proj_func in dir_projs.items():
      res["{}_{}".format(sensor, proj)] = sum(map(lambda p: fire(ship, p, sensor_func, proj_func), planets))
  
  return res


def my_ships_features(game_map, clusters):
  """Returns the features for each of our ships."""
  res = []
  for ship in game_map.get_me().all_ships():
    res.append(ship_features(ship, clusters, game_map.all_planets()))
  return res

#######################################################################
#### DEBUG ############################################################

def log_ship_features(ship, feats):
  res = []
  for key, value in feats.items():
    if value != 0.0:
      res.append((key, value))
  res.sort(key = lambda x: x[0])
  logging.info('\n'.join(["", str(ship.id)] + list(map(lambda x: "{}: {}".format(*x), res))))
