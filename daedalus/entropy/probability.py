#!/usr/bin/env python
"""
created 3/6/2024

@author Sharif Saleki

Description:
"""
from typing import Union
import numpy as np


def stay_signal(choices: Union[np.array, list]) -> np.array:
    """
    Computes a "stay" signal based on choices.

    Args:
        choices: input choices (list or np.array)

    Returns:
        stay: stay signal (np.array)

    """
    signal = np.array(choices)
    stay = np.equal(signal[1:], signal[:-1]).astype(int)

    return stay


def switch_signal(choices: Union[np.array, list]) -> np.array:
    """
    Computes a "switch" signal from an array of choices.

    Args:
        choices: input choices (list or np.array)

    Returns:
        switch: switch signal (np.array)

    """
    signal = np.array(choices)
    switch = np.not_equal(signal[1:], signal[:-1]).astype(int)

    return switch


def win_signal(rewards: Union[np.array, list]) -> np.array:
    """
    Win signal is the same as reward signal only indexed 0 to n-1

    Args:
        rewards: input signals (list or np.array)

    Returns:
        win: np.array
    """
    return np.array(rewards[:-1])


def lose_signal(rewards: Union[np.array, list]) -> np.array:
    """
    Lose signal is one minus win

    Args:
        rewards: input signals (list or np.array)

    Returns:
        lose: np.array
    """
    return 1 - np.array(rewards[:-1])


def handle_bad_probs(prob: float) -> float:
    """
    Handles bad values for probabilities

    Args:
        prob: float
            Probability

    Returns:
        prob: float
            Probability
    """
    if np.isclose(prob, 0) or np.isnan(prob) or np.isinf(prob) or np.isneginf(prob):
        prob = 1e-10

    return prob


def get_prob(x: Union[np.array, list]) -> float:
    """
    Calculates the probability of a given signal

    Args:
        x: input signal (list or np.array)

    Returns:
        prob: float
    """
    prob = np.mean(x)

    return handle_bad_probs(prob)


def joint_prob(x: Union[np.array, list], y: Union[np.array, list]) -> float:
    """
    Calculates the joint probability of two variables

    Args:
        x: first variable
        y: second variable

    Returns:
        prob: float
    """
    prob = np.mean(np.logical_and(x, y))

    return handle_bad_probs(prob)


def cond_prob(y: Union[np.array, list], x: Union[np.array, list], epsilon=1e-10) -> float:
    """
    Calculates conditional probability between two signals X and Y: P(Y|X) = P(X,Y) / P(X)

    Args:
        x: first signal (list or np.array)
        y: second signal (list or np.array)

    Returns:
        prob: conditional probability (float)
    """
    x = np.array(x)
    y = np.array(y)

    # check if mean of y is 0 or nan
    p_x = get_prob(x)
    p_x += epsilon
    prob = joint_prob(x, y) / p_x

    return handle_bad_probs(prob)


def three_way_prob(x: Union[np.array, list], y: Union[np.array, list], z: Union[np.array, list]) -> float:
    """
    Calculates the joint probability of three variables, using the chain rule:

        P(X,Y,Z) = P(X) * P(Y|X) * P(Z|X,Y)

    Args:
        x: first variable
        y: second variable
        z: third variable

    Returns:
        prob: float
    """
    p_x = get_prob(x)
    p_y_given_x = cond_prob(y, x)
    p_z_given_x_y = cond_prob(z, np.logical_and(x, y))

    return handle_bad_probs(p_x * p_y_given_x * p_z_given_x_y)


def calculate_kl_divergence(p, q, epsilon=1e-10):
    """
    Calculate the Kullback-Leibler divergence between two probability distributions.

    Args:
        p (np.array): First probability distribution.
        q (np.array): Second probability distribution.
        epsilon (float): Small value to add to probabilities to avoid log(0).

    Returns:
        float: The KL divergence D_KL(P || Q).
    """
    p = np.array(p)
    q = np.array(q)

    p = p + epsilon
    q = q + epsilon

    p = p / np.sum(p)
    q = q / np.sum(q)

    kl_divergence = np.sum(p * np.log(p / q))

    return kl_divergence
