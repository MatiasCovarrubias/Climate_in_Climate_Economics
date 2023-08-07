import tensorflow as tf
import numpy as np
import Parameters
import PolicyState
import State

# --------------------------------------------------------------------------- #
# Extract parameters
# --------------------------------------------------------------------------- #

# Economic parameters
beta, alpha, delta, rho, shock_sd = (
    Parameters.beta,
    Parameters.alpha,
    Parameters.delta,
    Parameters.rho,
    Parameters.shock_sd,
)

k_ss, i_ss, c_ss = Parameters.k_ss, Parameters.i_ss, Parameters.c_ss

# --------------------------------------------------------------------------- #
# Economic variables
# --------------------------------------------------------------------------- #


def Kplus(state, policy_state):
    """ Investment """
    return PolicyState.Kplus_norm(policy_state) * tf.math.exp(k_ss)


def consumption(state, policy_state):
    """ Consumption """
    return PolicyState.c_norm(policy_state) * tf.math.exp(c_ss)


def capital(state, policy_state):
    """ Capital """
    return tf.math.exp(State.k_norm(state) + k_ss)


def investment(state, policy_state):
    """ Investment """
    return Kplus(state, policy_state) - (1 - delta) * capital(state, policy_state)


def TFP(state, policy_state):
    """ TFP """
    return tf.math.exp(State.a_norm(state))


def ygross(state, policy_state):
    """ Gross output """
    _K = capital(state, policy_state)
    _A = TFP(state, policy_state)
    return _A * _K ** alpha
