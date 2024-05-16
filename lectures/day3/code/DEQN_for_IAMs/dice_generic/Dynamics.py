import tensorflow as tf
import math
import itertools
import State
import PolicyState
import Definitions
import Parameters


# --------------------------------------------------------------------------- #
# Deterministic case
# --------------------------------------------------------------------------- #

# Probability of a dummy shock
shock_probs = tf.constant([1.0])  # Dummy probability

def total_step_random(prev_state, policy_state):
    """ State dependant random shock to evaluate the expectation operator """
    _ar = AR_step(prev_state)
    _shock = shock_step_random(prev_state)
    _policy = policy_step(prev_state, policy_state)

    _total_random = _ar + _shock + _policy

    return _total_random


def total_step_spec_shock(prev_state, policy_state, shock_index):
    """ State specific shock to run one episode """
    _ar = AR_step(prev_state)
    _shock = shock_step_spec_shock(prev_state, shock_index)
    _policy = policy_step(prev_state, policy_state)

    _total_spec = _ar + _shock + _policy

    return _total_spec


def AR_step(prev_state):
    _ar_step = tf.zeros_like(prev_state)
    return _ar_step


def shock_step_random(prev_state):
    _shock_step = tf.zeros_like(prev_state)
    return _shock_step


def shock_step_spec_shock(prev_state, shock_index):
    _shock_step = tf.zeros_like(prev_state)
    return _shock_step


def policy_step(prev_state, policy_state):
    """ State variables are updated by the optimal policy (capital stock) or
    the laws of motion for carbon masses and temperatures """
    _policy_step = tf.zeros_like(prev_state)  # Initialization

    # Update state variables if needed
    _policy_step = State.update(
        _policy_step, 'kx', PolicyState.kplusy(policy_state))
    _policy_step = State.update(
        _policy_step, 'MATx', Definitions.MATplus(prev_state, policy_state))
    _policy_step = State.update(
        _policy_step, 'MUOx', Definitions.MUOplus(prev_state, policy_state))
    _policy_step = State.update(
        _policy_step, 'MLOx', Definitions.MLOplus(prev_state, policy_state))
    _policy_step = State.update(
        _policy_step, 'TATx', Definitions.TATplus(prev_state, policy_state))
    _policy_step = State.update(
        _policy_step, 'TOCx', Definitions.TOCplus(prev_state, policy_state))
    _policy_step = State.update(
        _policy_step, 'taux', Definitions.tau2tauplus(prev_state, policy_state)
    )

    return _policy_step
