from .filter import extended_kalman_filter
from .smoother import smoothing


def filtsmooth(x0, linear_transitions, continuous_observation_model):
    out, nll = extended_kalman_filter(
        x0, linear_transitions, continuous_observation_model
    )
    out = smoothing(linear_transitions, out)
    return out, nll
