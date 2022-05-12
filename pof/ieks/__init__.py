from .filtering import linear_noiseless_filtering
from .smoothing import smoothing


def filtsmooth(x0, linear_transitions, linear_observations):
    out, nll, _ = linear_noiseless_filtering(
        x0, linear_transitions, linear_observations
    )
    out, obj = smoothing(linear_transitions, out)
    return out, nll, obj
