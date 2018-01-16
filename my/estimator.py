import random, math
import numpy as np

import my.features as ft
from hlt.entity import Position


########################################################################
#### FEATURE EXPANSION #################################################

def identity(feats):
  """Returns only the basic feats, as a numpy array."""
  return np.array(list(map(lambda x: feats[x], ft.FEATURES)))


def get_nonmoves(feats):
  """Nonmoves are attributes which are not yet combined with one of the moves.
  This function calculates them from original features."""
  res = []
  for f in ft.FEATURES:
    res.append(feats[f])
  for sf in ft.STAT_FEATURES:
    for df in ft.DIR_FEATURES:
      res.append(feats[sf] * feats[df])
  return np.array(res)


def get_moves(feats):
  """Returns the movement features (up, down, left, right) as a numpy array."""
  dx = feats.get("dx", 0.0)
  dy = feats.get("dy", 0.0)
  res = []
  for d in ft.DIRECTIONS:
    res.append(ft.fire(Position(0, 0), Position(dx, dy), ft.distance, ft.dir_projs[d]))
  return np.array(res)


def fight_expand(feats):
  """Creates new features by combining in various ways original features
  and the direction of movement."""
  dx = feats.get("dx", 0.0)
  dy = feats.get("dy", 0.0)
  
  moves = get_moves(feats)
  nonmoves = get_nonmoves(feats)
  
  # In addition to nonmoves, add nonmoves combined with moves.
  res = []
  res.extend(nonmoves)
  for val1 in nonmoves:
    for val2 in moves:
      res.append(val1 * val2)
  
  return np.array(res)

########################################################################
#### MOVE MAKER ########################################################

def fight(feats, fight_estimator):
  """Tries out several moves, and chooses the one with the highest predicted value."""
  res = (0, 0)
  best = -math.inf
  
  # Try various movement commands.
  for t in range(99):
    angle = random.randint(0, 359)
    phi = math.radians(angle)
    speed = random.randint(random.randint(0, 7), 7)
    dx = speed * math.cos(phi)
    dy = speed * math.sin(phi)
    feats["dx"] = dx
    feats["dy"] = dy
    val = fight_estimator.value_of(feats)
    if val > best:
      res = (speed, angle)
      best = val
  
  return res

########################################################################
#### ESTIMATOR (wrapper for predictor) #################################

class Estimator:
  def __init__(self, model, expander = identity):
    self.expander = expander
    self.model = model
  
  def value_of(self, feats):
    x = self.expander(feats)
    x = x.reshape(1, x.shape[0])
    return float(self.model.predict(x))
