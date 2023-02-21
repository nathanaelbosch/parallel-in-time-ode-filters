from .filter import extended_kalman_filter, linear_noiseless_filter
from .smoother import smoothing


def filtsmooth(x0, linear_transitions, continuous_observation_model):
    out, nll, ssq = extended_kalman_filter(
        x0, linear_transitions, continuous_observation_model
    )
    out, obj = smoothing(linear_transitions, out)
    return out, nll, obj, ssq


def linear_filtsmooth(x0, linear_transitions, linear_observations):
    out, nll, ssq = linear_noiseless_filter(x0, linear_transitions, linear_observations)
    out, obj = smoothing(linear_transitions, out)
    return out, nll, obj, ssq
