from .filtering import linear_noiseless_filtering
from .smoothing import smoothing


def filtsmooth(x0, linear_transitions, linear_observations):
    out = linear_noiseless_filtering(x0, linear_transitions, linear_observations)
    return smoothing(linear_transitions, out)
