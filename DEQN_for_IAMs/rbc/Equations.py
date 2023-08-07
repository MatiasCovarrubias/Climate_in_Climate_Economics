import tensorflow as tf
import Definitions
import PolicyState
import Parameters
import State


def equations(state, policy_state):
    """ The dictionary of loss functions """
    # Expectation operator
    E_t = State.E_t_gen(state, policy_state)

    # ----------------------------------------------------------------------- #
    # Parameters
    # ----------------------------------------------------------------------- #
    beta, alpha, delta = (
        Parameters.beta,
        Parameters.alpha,
        Parameters.delta,
    )

    # policy
    C = Definitions.consumption(state, policy_state)

    # economic variables
    Y = Definitions.ygross(state, policy_state)
    Inv = Definitions.investment(state, policy_state)

    # ----------------------------------------------------------------------- #
    # Loss functions
    # ----------------------------------------------------------------------- #
    loss_dict = {}

    loss_dict["euler_residual"] = tf.reduce_mean(
        (
            1
            - (
                beta
                * E_t(
                    lambda s, ps: (Definitions.consumption(s, ps) / C) ** (-1)
                    * (
                        1
                        - delta
                        + alpha
                        * Definitions.TFP(s, ps)
                        * Definitions.capital(s, ps) ** (alpha - 1)
                    )
                )
            )
        )
        ** 2
    )

    loss_dict["market_clearing"] = tf.reduce_mean((Y - C - Inv) ** 2)

    return loss_dict
