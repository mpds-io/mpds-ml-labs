
# These are fully empirical, estimated using ml_knn and
# kept in testing purposes, not to generate nonsense
prediction_ranges = {
    'z': [39, 284],
    'y': [-279, -27],
    'x': [13, 24],
    'k': [-92, 245],
    'w': [1.5, 7.7],
    'm': [72, 2594],
    'd': [159, 999],
    't': [1.1, 43.0],
    'i': [-17, 12],
    'o': [7, 106]
}

# a part of the prop max-min distance, as a tolerance,
# for estimating the prediction quality;
# the more this distance, the more bad predictions are considered as good ones
RANGE_TOLERANCE = 0.30

# if no knn results found,
# how far beyond are we allowed
# to approximate in a knn query
prediction_margins = {prop_id: (bounds[1] - bounds[0]) / 4 for prop_id, bounds in prediction_ranges.items()}
for prop_id in ['x', 'w', 't']:
    prediction_margins[prop_id] *= 10
prediction_margins['i'] *= 100
