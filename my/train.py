import argparse
import json
import os.path, subprocess
import zipfile
import itertools

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam

from my.data import to_table, get_Xy
from my.estimator import fight_expand, identity, Estimator


########################################################################
#### Methods for loading replays from directories/zips. ################
#### Taken from the ML-StarterBot's train module. ######################

def fetch_data_dir(directory, limit):
  """
  Loads up to limit games into Python dictionaries from uncompressed replay files,
  and yield them one by one.
  """
  replay_files = sorted([f for f in os.listdir(directory) if
                         os.path.isfile(os.path.join(directory, f)) and f.startswith("replay-")])
  
  if len(replay_files) == 0:
    raise Exception("Didn't find any game replays. Please call make games.")
  
  print("Found {} games.".format(len(replay_files)))
  print("Trying to load up to {} games ...".format(limit))
  
  for counter, r in enumerate(replay_files):
    print("Game number {}".format(counter + 1))
    full_path = os.path.join(directory, r)
    with open(full_path) as game:
      game_data = game.read()
      game_json_data = json.loads(game_data)
      yield(game_json_data)
    if counter >= limit:
      break


def fetch_data_zip(zipfilename, limit):
    """
    Loads up to limit games into Python dictionaries from a zipfile containing uncompressed replay files,
    and yield them one by one.
    """
    with zipfile.ZipFile(zipfilename) as z:
        print("Found {} games.".format(len(z.filelist)))
        print("Trying to load up to {} games ...".format(limit))
        for counter, i in enumerate(z.filelist[:limit]):
            print("Game number {}".format(counter + 1))
            with z.open(i) as f:
                lines = f.readlines()
                assert len(lines) == 1
                d = json.loads(lines[0].decode())
                yield(d)

########################################################################
#### ML HELPERS ########################################################

def validate(X, y, fitter, k = 10):
  """How good is our model? Use k-fold cross validation."""
  print("Validating...")
  n = len(y)
  
  total_train_error = 0.0
  total_val_error = 0.0
  for i in range(k):
    print("step {}/{}".format(i + 1, k))
    l = i*n // k
    r = (i+1)*n // k
    train_indices = list(range(l)) + list(range(r, n))
    val_indices = list(range(l, r))
    
    # Train it on the selected training data.
    tX = X[train_indices, :]
    ty = y[train_indices]
    estimator = fitter.fit(tX, ty)
    
    # Compute training error.
    p_ty = estimator.predict(tX)
    total_train_error += np.mean((p_ty - ty)**2)
    
    # Compute validation error.
    vX = X[val_indices, :]
    vy = y[val_indices]
    p_vy = estimator.predict(vX)
    total_val_error += np.mean((p_vy - vy)**2)
    
    print("Training error:", total_train_error / (i+1))
    print("Validation error:", total_val_error / (i+1))

########################################################################
#### MODEL LEARNING ####################################################

def learn_regression(X, y, src = None, save_location = None, verbose = True):
  """Fits linear regression on the given data, saves the model into <save_location>.
  If <verbose>, computes training and validation errors. If <src> is given,
  instead of starting from scratch starts from there."""
  if src is None:
    src = LinearRegression(normalize = True)
  if verbose:
    validate(X, y, src)
  model = src.fit(X, y)
  if save_location is not None:
    joblib.dump(model, save_location)
  return model


def learn_neural_net(X, y, src = None, save_location = None, verbose = True):
  """Fits neural network on the given data, saves the model into <save_location>.
  If <verbose>, the training process will print some info. If <src> is given,
  instead of starting from scratch starts from there."""
  mlp = src
  if mlp is None:
    mlp = Sequential()
    mlp.add(Dense(100, activation="tanh", input_dim = X.shape[1]))
    mlp.add(Dense(1, activation="linear"))
    mlp.compile(loss = "mse", optimizer = SGD(lr = 0.000004))
  mlp.fit(X, y, epochs = 100, validation_split = 0.1, verbose = verbose)
  if save_location is not None:
    mlp.save(save_location)
  return mlp


def self_play(estimator, learn, save_location = None, epochs = 10, min_rows = 4 * 10**4):
  """Runs <epochs> training sessions. In each one, we let the bot play
  with itself until enough data is gathered, and then we have him learn
  on that data."""
  
  # Get the next directory for self_play.
  directory = "self_play"
  start = 0
  while os.path.isdir(os.path.join(directory, str(start))):
    start += 1
  
  for epoch in range(start, start + epochs):
    print("Epoch", epoch)
    print("------------------------------------------------------")
    num_rows = 0
    subtables = []
    
    epoch_dir = os.path.join(directory, str(epoch))
    subprocess.run("mkdir {}".format(epoch_dir), shell = True)
    
    # Have a few games until our table is large enough.    
    for game_num in itertools.count():
      print("Epoch", epoch, "game", game_num)
      print("---------------------------")
      
      game_dir = os.path.join(epoch_dir, str(game_num))
      subprocess.run("mkdir {}".format(game_dir), shell = True)
      
      command = ["./halite", "--no-compression", "-t", "-i {}".format(game_dir)] + ["python3 MyBot_random.py"]*4
      subprocess.run(command)
      
      # Process the created replay.
      for data in fetch_data_dir(game_dir, 1):
        sub = to_table(data, 1.0)
        num_rows += sub.shape[0]
        subtables.append(sub)
      
      # Break if we have enough data.
      if num_rows >= min_rows:
        print("Enough data: {}".format(num_rows))
        break
      else:
        print("Not enough data: {}".format(num_rows))
    
    table = np.concatenate(subtables)
    csv_loc = os.path.join(epoch_dir, "dump.csv")
    np.savetxt(csv_loc, table, delimiter = ',')
    
    X, y = get_Xy(table, estimator.expander)
    estimator.model = learn(X, y, estimator.model, save_location)
  
  return estimator

########################################################################
#### TRAINING PROCEDURE ################################################

def main():  
  parser = argparse.ArgumentParser(description="ML-Individual training")
  parser.add_argument("--data", help = "Data directory or zip file containing uncompressed games")
  parser.add_argument("--games_limit", type=int, help="Train on up to games_limit games", default = 100)
  parser.add_argument("--dump_location", help="Location where processed data should be stored", default = "dump.csv")
  parser.add_argument("--model_location", help="Directory where model should be stored", default = "model")
  parser.add_argument("--sample_ratio", type=float, help="Percentage of frames should we take from each game.", default = 0.1)
  parser.add_argument("--discount", type=float, help="MDP model: discount factor.", default = 0.9)
  parser.add_argument("--max_len", type=int, help="MDP model: how far into the future do we see when calculating utilities.", default = 50)
  parser.add_argument("--sp_eps", type=int, help="Number of epochs in self_play. If 0 (default), instead learns from given data.", default = 0)
  parser.add_argument("--sp_rows", type=int, help="The number of rows in the table during self-play that is considered 'enough'.", default = 4 * 10**4)
  parser.add_argument("--learner", help="Which learner do we employ? (0: linear, 1: neural_net)", default = "neural_net")
  
  args = parser.parse_args()
  
  if args.sp_eps > 0:
    # Self play.
    if args.learner == "linear":
      estimator = Estimator(None, fight_expand)
      estimator = self_play(estimator, learn_regression, "model/regressor.pkl", args.sp_eps, args.sp_rows)
    elif args.learner == "neural_net":
      estimator = Estimator(None, identity)
      estimator = self_play(estimator, learn_neural_net, "model/neural_net.h5", args.sp_eps, args.sp_rows)
  
  else:
    if args.data.endswith('.csv'):
      # Load the stored data.
      table = np.loadtxt(args.data, delimiter = ',')
    else:
      # Load the raw game data.
      if args.data.endswith('.zip'):
        raw_data = fetch_data_zip(args.data, args.games_limit)
      else:
        raw_data = fetch_data_dir(args.data, args.games_limit)
      
      # Process all the data and store it somewhere.
      table = np.concatenate(tuple(map(lambda x: to_table(x, args.sample_ratio, args.discount, args.max_len), raw_data)))
      np.savetxt(args.dump_location, table, delimiter = ',')
    
    if args.learner == "linear":
      X, y = get_Xy(table, fight_expand)
      learn_regression(X, y, save_location = "model/regressor.pkl")
    elif args.learner == "neural_net":
      X, y = get_Xy(table)
      learn_neural_net(X, y, save_location = "model/neural_net.h5")


if __name__ == "__main__":
  main()
