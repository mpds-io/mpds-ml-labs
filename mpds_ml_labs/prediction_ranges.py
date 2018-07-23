
# This is fully empirical,
# kept in testing purposes
# not to generate nonsense

prediction_ranges = {
    'z': [5, 265],
    'y': [-325, 0],
    'x': [11, 28],
    'k': [-150, 225],
    'w': [0.5, 10],
    'm': [300, 2700],
    'd': [175, 1100],
    't': [-0.5, 10]
}

TOL_QUALITY = 0.30 # a part of the prop max-min distance, as a tolerance,
                   # for estimating the prediction quality;
                   # the more this distance, the more bad predictions are considered as good ones
