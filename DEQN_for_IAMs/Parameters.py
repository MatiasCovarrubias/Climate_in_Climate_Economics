#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 09:07:14 2020

@author: -
"""

import tensorflow as tf
import hydra
import os
import sys

#### Configuration setup
@hydra.main(config_path="config", config_name="config")
# @hydra.main(config_path=(os.environ["USE_CONFIG_FROM_RUN_DIR"] + "/.hydra/config.yaml") if "USE_CONFIG_FROM_RUN_DIR" in os.environ.keys() else "config/config.yaml")
def set_conf(cfg):
    print(cfg.pretty())

    # the model we are running
    setattr(sys.modules[__name__], "MODEL_NAME", cfg.MODEL_NAME)

    # RNG
    tf.random.set_seed(cfg.seed)
    rng_state = tf.Variable([0, 0, cfg.seed], dtype=tf.int64)
    setattr(
        sys.modules[__name__],
        "rng",
        tf.random.Generator.from_state(rng_state, alg="philox"),
    )

    # RUN CONFIGURATION
    setattr(sys.modules[__name__], "N_sim_batch", cfg.N_sim_batch)
    setattr(sys.modules[__name__], "N_epochs_per_episode", cfg.N_epochs_per_episode)
    setattr(sys.modules[__name__], "N_minibatch_size", cfg.N_minibatch_size)
    setattr(sys.modules[__name__], "N_episode_length", cfg.N_episode_length)
    setattr(sys.modules[__name__], "N_episodes", cfg.N_episodes)
    setattr(
        sys.modules[__name__],
        "expectation_pseudo_draws",
        cfg.get("expectation_pseudo_draws", 5),
    )
    setattr(
        sys.modules[__name__],
        "expectation_type",
        cfg.get("expectation_type", "product"),
    )

    # VARIABLES
    setattr(sys.modules[__name__], "states", [s["name"] for s in cfg.states])
    setattr(sys.modules[__name__], "policy_states", [s["name"] for s in cfg.policies])
    setattr(sys.modules[__name__], "definitions", [s["name"] for s in cfg.definitions])

    # OUTPUT FILE FOR ERROR MEASURES
    setattr(sys.modules[__name__], "error_filename", cfg.error_filename)

    state_bounds = {"lower": {}, "penalty_lower": {}, "upper": {}, "penalty_upper": {}}

    for s in cfg.states:
        if "bounds" in s.keys() and "lower" in s["bounds"].keys():
            state_bounds["lower"][s["name"]] = s["bounds"]["lower"]
            if "penalty_lower" in s["bounds"].keys():
                penalty = s["bounds"]["penalty_lower"]
            else:
                penalty = 1 / s["bounds"]["lower"] ** 2
            state_bounds["penalty_lower"][s["name"]] = penalty

        if "bounds" in s.keys() and "upper" in s["bounds"].keys():
            state_bounds["upper"][s["name"]] = s["bounds"]["upper"]
            if "penalty_upper" in s["bounds"].keys():
                penalty = s["bounds"]["penalty_upper"]
            else:
                penalty = 1 / s["bounds"]["upper"] ** 2
            state_bounds["penalty_upper"][s["name"]] = penalty

    setattr(sys.modules[__name__], "state_bounds_hard", state_bounds)

    policy_bounds = {"lower": {}, "penalty_lower": {}, "upper": {}, "penalty_upper": {}}

    for s in cfg.policies:
        if "bounds" in s.keys() and "lower" in s["bounds"].keys():
            policy_bounds["lower"][s["name"]] = s["bounds"]["lower"]
            if "penalty_lower" in s["bounds"].keys():
                penalty = s["bounds"]["penalty_lower"]
            else:
                penalty = 1 / s["bounds"]["lower"] ** 2
            policy_bounds["penalty_lower"][s["name"]] = penalty

        if "bounds" in s.keys() and "upper" in s["bounds"].keys():
            policy_bounds["upper"][s["name"]] = s["bounds"]["upper"]
            if "penalty_upper" in s["bounds"].keys():
                penalty = s["bounds"]["penalty_upper"]
            else:
                penalty = 1 / s["bounds"]["upper"] ** 2
            policy_bounds["penalty_upper"][s["name"]] = penalty

    setattr(sys.modules[__name__], "policy_bounds_hard", policy_bounds)

    definition_bounds = {
        "lower": {},
        "penalty_lower": {},
        "upper": {},
        "penalty_upper": {},
    }

    for s in cfg.definitions:
        if "bounds" in s.keys() and "lower" in s["bounds"].keys():
            definition_bounds["lower"][s["name"]] = s["bounds"]["lower"]
            if "penalty_lower" in s["bounds"].keys():
                penalty = s["bounds"]["penalty_lower"]
            else:
                penalty = 1 / s["bounds"]["lower"] ** 2
            definition_bounds["penalty_lower"][s["name"]] = penalty

        if "bounds" in s.keys() and "upper" in s["bounds"].keys():
            definition_bounds["upper"][s["name"]] = s["bounds"]["upper"]
            if "penalty_upper" in s["bounds"].keys():
                penalty = s["bounds"]["penalty_upper"]
            else:
                penalty = 1 / s["bounds"]["upper"] ** 2
            definition_bounds["penalty_upper"][s["name"]] = penalty

    setattr(sys.modules[__name__], "definition_bounds_hard", definition_bounds)

    # NEURAL NET
    tf.keras.backend.set_floatx(cfg.get("keras_precision", "float32"))

    layers = []

    for i, layer in enumerate(cfg.layers, start=1):
        if i < len(cfg.layers):
            if "dropout_rate" in layer["hidden"]:
                layers.append(
                    tf.keras.layers.Dropout(rate=layer["hidden"]["dropout_rate"])
                )
            if "batch_normalize" in layer["hidden"]:
                layers.append(
                    tf.keras.layers.BatchNormalization(
                        **layer["hidden"]["batch_normalize"]
                    )
                )
            layers.append(
                tf.keras.layers.Dense(
                    units=layer["hidden"]["units"],
                    activation=layer["hidden"]["activation"],
                    kernel_initializer=tf.keras.initializers.VarianceScaling(
                        scale=layer["hidden"].get("init_scale", 1.0),
                        mode=cfg.get("net_initializer_mode", "fan_in"),
                        distribution=cfg.get(
                            "net_initializer_distribution", "truncated_normal"
                        ),
                        seed=i,
                    ),
                )
            )
        else:
            layers.append(
                tf.keras.layers.Dense(
                    units=len(policy_states),
                    activation=layer["output"]["activation"],
                    kernel_initializer=tf.keras.initializers.VarianceScaling(
                        scale=layer["output"].get("init_scale", 1.0),
                        mode=cfg.get("net_initializer_mode", "fan_in"),
                        distribution=cfg.get(
                            "net_initializer_distribution", "truncated_normal"
                        ),
                        seed=i,
                    ),
                )
            )

    policy_net = tf.keras.models.Sequential(layers)
    policy_net.build(input_shape=(None, len(states)))

    # apply post-processing per-variable
    def policy(s):
        raw_policy = policy_net(s)
        for i, pol in enumerate(cfg.policies):
            if "activation" in pol.keys():
                activation_str = pol["activation"]
                if pol["activation"] == "implied":
                    if (
                        "lower" in pol["bounds"].keys()
                        and "upper" in pol["bounds"].keys()
                    ):
                        activation_str = (
                            "lambda x: {l} + ({u} - {l}) * tf.math.sigmoid(x)".format(
                                l=str(pol["bounds"]["lower"]),
                                u=str(pol["bounds"]["upper"]),
                            )
                        )
                raw_policy = tf.tensor_scatter_nd_update(
                    raw_policy,
                    [[j, i] for j in range(s.shape[0])],
                    eval(activation_str)(raw_policy[:, i]),
                )

        if cfg.keras_precision == "float64":
            return tf.cast(raw_policy, tf.dtypes.float32)

        return raw_policy

    setattr(sys.modules[__name__], "policy", policy)
    setattr(sys.modules[__name__], "policy_net", policy_net)

    # OPTIMIZER
    setattr(
        sys.modules[__name__],
        "optimizer",
        getattr(tf.keras.optimizers, cfg.optimizer)(
            learning_rate=cfg.learning_rate, clipvalue=cfg.clipvalue
        ),
    )

    # CONSTANTS
    for (key, value) in cfg.constants.items():
        setattr(sys.modules[__name__], key, value)

    # STATE INITIALIZATION
    def initialize_states():
        # starting state
        init_val = tf.ones([N_sim_batch, len(states)])
        # apply special inits if any
        for i, s in enumerate(cfg.states):
            if "init" in s:
                init_val = tf.tensor_scatter_nd_update(
                    init_val,
                    [[j, i] for j in range(init_val.shape[0])],
                    getattr(rng, s["init"]["distribution"])(
                        shape=(N_sim_batch,), **s["init"]["kwargs"]
                    ),
                )
        return init_val

    starting_state = tf.Variable(initialize_states())

    setattr(sys.modules[__name__], "starting_state", starting_state)
    setattr(sys.modules[__name__], "initialize_states", initialize_states)
    setattr(
        sys.modules[__name__],
        "initialize_each_episode",
        cfg.get("initialize_each_episode", False),
    )

    # LOGGING
    setattr(sys.modules[__name__], "LOG_DIR", os.getcwd())

    if cfg.STARTING_POINT == "NEW":
        for file in os.scandir(os.getcwd()):
            if not ".hydra" in file.path:
                os.unlink(file.path)

    setattr(sys.modules[__name__], "writer", tf.summary.create_file_writer(os.getcwd()))

    setattr(sys.modules[__name__], "current_episode", tf.Variable(1))
    ckpt = tf.train.Checkpoint(
        step=tf.Variable(1),
        current_episode=current_episode,
        optimizer=optimizer,
        policy=policy_net,
        rng_state=rng_state,
        starting_state=starting_state,
    )
    manager = tf.train.CheckpointManager(
        ckpt,
        os.getcwd(),
        max_to_keep=cfg.MAX_TO_KEEP_NUMBER,
        step_counter=current_episode,
        checkpoint_interval=cfg.CHECKPOINT_INTERVAL,
    )

    if cfg.STARTING_POINT == "LATEST" and manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
        ckpt.restore(manager.latest_checkpoint)

    if cfg.STARTING_POINT != "LATEST" and cfg.STARTING_POINT != "NEW":
        print("Restored from {}".format(cfg.STARTING_POINT))
        ckpt.restore(cfg.STARTING_POINT)

    setattr(sys.modules[__name__], "ckpt", ckpt)
    setattr(sys.modules[__name__], "manager", manager)

    tf.print("Optimizer configuration:")
    tf.print(optimizer.get_config())

    tf.print("Starting state:")
    tf.print(starting_state)


set_conf()
