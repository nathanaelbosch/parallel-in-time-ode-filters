from .filter import linear_noiseless_filtering
from .smoother import smoothing


def linear_filtsmooth(x0, linear_transitions, linear_observations):
    out, nll, _, ssq = linear_noiseless_filtering(
        x0, linear_transitions, linear_observations
    )
    out, obj = smoothing(linear_transitions, out)
    return out, nll, obj, ssq
