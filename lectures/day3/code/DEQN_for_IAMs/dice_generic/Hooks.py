from Parameters import policy, states, policy_states
import PolicyState
import State
import tensorflow as tf
import Definitions
from Parameters import definitions, definition_bounds_hard, MODEL_NAME


def cycle_hook(state, i):
    policy_state = policy(state)
    for s in states:
        tf.summary.histogram(
            "hist_" + s, getattr(State, s)(state), step=i)

    for p in policy_states:
        tf.summary.histogram(
            "hist_" + p, getattr(PolicyState, p)(policy_state), step=i)

    for d in definitions:
        tf.summary.histogram(
            "hist_" + d, getattr(Definitions, d)(state, policy_state), step=i)

    return True
