import numpy as np
from sklearn.cluster import KMeans
from hlt.entity import Entity, Position

from matplotlib import pyplot as plt
import logging


class Cluster(Entity):
  """Class representing a cluster of ships."""
  
  def __init__(self, x, y, ships):
    self.x = x
    self.y = y
    self.ships = ships
    self.radius = (0.0 if len(ships) == 0 else max([self.calculate_distance_between(ship) for ship in ships]))
    self.health = sum(map(lambda s: s.health, ships))
    self.owner = (None if len(ships) == 0 else ships[0].owner)
    self.id = None
  
  def dist_min(self, pos):
    return max(0.0, self.calculate_distance_between(pos) - self.radius)
  
  def dist_max(self, pos):
    return self.calculate_distance_between(pos) + self.radius
  
  def dist_geo(self, pos):
    return (self.dist_min(pos) * self.dist_max(pos))**0.5
  
  @property
  def size(self):
    return len(self.ships)


def get_clusters(ships, k = 60):
  """Divide <ships> into <k> clusters and return a list of those clusters."""
  k = min(k, len(ships))
  if k == 0:
    return []
  
  # Calculate the k clusters, divide ships based on their label.
  ship_array = np.array([[s.x, s.y] for s in ships])
  kmeans = KMeans(n_clusters = k).fit(ship_array)
  followers = [[] for i in range(k)]
  for i, label in np.ndenumerate(kmeans.labels_):
    followers[label].append(ships[i[0]])
  
  # Create the clusters and return them.
  res = []
  for i in range(k):
    if len(followers[i]) == 0:
      continue
    cx, cy = kmeans.cluster_centers_[i]
    res.append(Cluster(cx, cy, followers[i]))
  return res


def all_clusters(game_map, k_fighters = 60, k_miners = 30):
  """Divide all ships into clusters based on their owner and whether they
  are fighters (free to do stuff) or miners."""
  clusters = []
  for player in game_map.all_players():
    fighters = []
    miners = []
    for ship in player.all_ships():
      if ship.docking_status == ship.DockingStatus.UNDOCKED:
        fighters.append(ship)
      else:
        miners.append(ship)
    pc = {"fighters": get_clusters(fighters, k_fighters), "miners": get_clusters(miners, k_miners)}
    clusters.append(pc)
  return clusters


#######################################################################
#### DEBUG ############################################################

curr_img_id = 0

def snapshot(clusters, img_name):
  global curr_img_id
  ships = []
  for i, cluster in enumerate(clusters):
    for ship in cluster.ships:
      ships.append([ship.x, ship.y, i])
  ships_array = np.array(ships)
  plt.scatter(ships_array[:, 0], ships_array[:, 1], c = ships_array[:, 2])
  plt.savefig("{}_{}.png".format(img_name, curr_img_id))
  plt.gcf().clear()
  curr_img_id += 1
