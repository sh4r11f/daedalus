#!/usr/bin/env python
"""
created 3/6/2024

@author Sharif Saleki

Description:
"""
import numpy as np

from .probability import get_prob, joint_prob, cond_prob


def entropy(x, epsilon=1e-10):
    """
    Calculates the entropy of a signal H(X) = -Σ p(x) log p(x)

    Args:
        x: input signal

    Returns:
        entropy
    """
    unique_vals = np.unique(x)
    h = 0

    for val in unique_vals:
        val_array = np.where(x == val, 1, 0)
        p = get_prob(val_array)
        p += epsilon
        h -= p * np.log2(p)

    return h


def joint_entropy(x, y, epsilon=1e-10):
    """
    Calculate the joint entropy of two random variables, X and Y. H(X,Y) = -Σ p(x,y) log p(x,y)

    Args:
        x (np.array): Array containing the values of random variable X.
        y (np.array): Array containing the values of random variable Y.

    Returns:
        float: joint entropy.
    """
    x_unique = np.unique(x)
    y_unique = np.unique(y)

    h = 0
    for x_val in x_unique:
        x_array = np.where(x == x_val, 1, 0)
        for y_val in y_unique:
            y_array = np.where(y == y_val, 1, 0)

            jp = joint_prob(x_array, y_array)
            jp += epsilon
            h -= jp * np.log2(jp)

    return h


def cond_entropy(y, x, epsilon=1e-10):
    """
    Calculate the conditional entropy of random variable Y given random variable X.
        H(Y|X) = -Σ p(x,y) log p(x,y)/p(x) = -Σ p(x,y) log p(y|x)

        also H(Y|X) = H(X,Y) - H(X) = H(Y) - I(X;Y)

    Args:
        y: Array containing the values of random variable Y.
        x: Array containing the values of random variable X.

    Returns:
        Conditional entropy value.
    """
    x_unique = np.unique(x)
    y_unique = np.unique(y)
    h = 0

    for x_val in x_unique:
        x_array = np.where(x == x_val, 1, 0)
        for y_val in y_unique:
            y_array = np.where(y == y_val, 1, 0)

            jp = joint_prob(x_array, y_array)
            cp = cond_prob(y_array, x_array)

            cp += epsilon
            h -= jp * np.log2(cp)

    return h


def mutual_information(x, y):
    """
    Calculates the mutual information between two signals.

    I(X;Y) = H(Y) - H(Y|X)

    Args:
        x: input signal
        y: input signal

    Returns:
        mutual information
    """
    h_y = entropy(y)
    h_y_given_x = cond_entropy(y, x)
    mi = h_y - h_y_given_x

    return mi


def mutual_information_direct(x, y, epsilon=1e-10):
    """
    Calculate mutual information between two discrete variables.

    Args:
        x: Array containing discrete values of the first variable.
        y: Array containing discrete values of the second variable.

    Returns:
        Mutual information value.
    """
    # Calculate joint probability distribution
    joint_prob = np.zeros((len(np.unique(x)), len(np.unique(y))))

    for i, val_x in enumerate(np.unique(x)):
        for j, val_y in enumerate(np.unique(y)):
            joint_prob[i, j] = np.sum((x == val_x) & (y == val_y)) / len(x)

    # Calculate marginal probabilities
    marginal_prob_x = np.sum(joint_prob, axis=1)
    marginal_prob_y = np.sum(joint_prob, axis=0)

    # Calculate mutual information
    mi = 0.0
    for i, val_x in enumerate(np.unique(x)):
        for j, val_y in enumerate(np.unique(y)):
            p_xy = joint_prob[i, j]
            p_x = marginal_prob_x[i]
            p_y = marginal_prob_y[j]

            p_x += epsilon
            p_y += epsilon

            mi += p_xy * np.log2(p_xy / (p_x * p_y))

    return mi
