import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from keras.models import load_model

from my.bot import Bot
from my.estimator import Estimator, identity


model = load_model("model/neural_net.h5")
estimator = Estimator(model, identity)

Bot(estimator, "Neural Net").play()
