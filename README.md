## Halite II player

### Training

To train a neural net model on `dump.csv`, run the following command:

`python3 -m my.train --data dump.csv`

The trained model will be stored in `model/neural_net.h5`.
Alternatively, if you want to do linear regression instead, add the `--learn linear` argument, and the trained model will be stored in `model/regressor.pkl`.

If you want to train the model by self-play, include the following two arguments: `--sp_eps <number>` and `--sp_rows <number>`. The former determines the number of training epochs, and the latter determines the amount of data required per epoch. More concretely, self-play works as follows: we let the current bot play games. After each game, we process the replay file and append the processed data to the current epoch's table. Then, if the table is large enough, we stop the current epoch, and train the bot on the gathered data. The trained bot is used in the next epoch.

Self-play currently works only with the neural net bot. (Not that it would make any difference... it still doesn't learn anything.)

### Running a game

If you want to run a game consisting of 4 random players, run the `run_randoms.sh` script. For a game of 4 neural net players, run `run_neurals.sh`. The replay of the game will be stored in the same directory, and can be viewed at [](https://halite.io/play-programming-challenge).

If you have problem with the provided `halite` binary file, you can download one of the [starter kits](https://halite.io/learn-programming-challenge/downloads-and-starter-kits/) which come together with better suited binary file.
