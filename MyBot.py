from sklearn.externals import joblib

from my.bot import Bot
from my.estimator import Estimator, fight_expand


model = joblib.load("model/regressor.pkl")
estimator = Estimator(model, fight_expand)

Bot(estimator, "Regressor").play()
