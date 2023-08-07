import tensorflow as tf
import State
import Definitions
import Parameters

# Extract parameters

beta, alpha, delta, rho, shock_sd = (
    Parameters.beta,
    Parameters.alpha,
    Parameters.delta,
    Parameters.rho,
    Parameters.shock_sd,
)

k_ss, i_ss, c_ss = Parameters.k_ss, Parameters.i_ss, Parameters.c_ss

# --------------------------------------------------------------------------- #
# Deterministic case
# --------------------------------------------------------------------------- #

# Probability of a dummy shock
shock_values = tf.constant([-1.2816, -0.6745, 0, 0.6745, 1.2816])  # Dummy shock
shock_probs = tf.constant([0.2, 0.2, 0.2, 0.2, 0.2])  # Dummy probability


def total_step_random(prev_state, policy_state):
    """ State dependant random shock to evaluate the expectation operator """
    Kplus = Definitions.Kplus(prev_state, policy_state)
    kplus_norm = tf.math.log(Kplus) - k_ss
    a = State.a_norm(prev_state)
    # Drawing a sample
    sample_index = tf.random.categorical(tf.math.log([shock_probs]), num_samples=1)

    # Getting the corresponding shock value
    sample_shock = tf.gather(shock_values, sample_index)

    # Removing extra dimensions
    sample_shock = tf.squeeze(sample_shock)
    aplus_norm = rho * a + shock_sd * sample_shock

    # Updating the state
    _total_random = tf.zeros_like(prev_state)
    _total_random = State.update(_total_random, "a_norm", aplus_norm)
    _total_random = State.update(_total_random, "k_norm", kplus_norm)

    return _total_random


def total_step_spec_shock(prev_state, policy_state, shock_index):
    """ State specific shock to run one episode """

    Kplus = Definitions.Kplus(prev_state, policy_state)
    kplus_norm = tf.math.log(Kplus) - k_ss
    a = State.a_norm(prev_state)
    shock = tf.gather(shock_values, shock_index)
    shock = tf.squeeze(shock)
    aplus_norm = rho * a + shock_sd * shock

    # Updating the state
    _total_spec = tf.zeros_like(prev_state)
    _total_spec = State.update(_total_spec, "a_norm", aplus_norm)
    _total_spec = State.update(_total_spec, "k_norm", kplus_norm)

    return _total_spec
