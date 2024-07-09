#!/usr/bin/env python
"""
created 3/6/2024

@author Sharif Saleki

Description:
"""
import numpy as np
from .probability import (
    stay_signal, switch_signal, win_signal, lose_signal,
    joint_prob, cond_prob, three_way_prob,
)
from scipy.ndimage import generic_filter


class BaseMetrics:
    """
    """
    def __init__(self, choices, rewards):

        self._choices = np.array(choices)
        self._rewards = np.array(rewards)

        self.stay_signal = stay_signal(choices)
        self.switch_signal = switch_signal(choices)
        self.win_signal = win_signal(rewards)
        self.lose_signal = lose_signal(rewards)

    @property
    def choices(self):
        return self._choices

    @property
    def rewards(self):
        return self._rewards

    @choices.setter
    def choices(self, choices):
        self.stay_signal = stay_signal(choices)
        self.switch_signal = switch_signal(choices)
        self._choices = np.array(choices)

    @rewards.setter
    def rewards(self, rewards):
        self.win_signal = win_signal(rewards)
        self.lose_signal = lose_signal(rewards)
        self._rewards = np.array(rewards)


class ProbabilityMetrics(BaseMetrics):
    """
    """
    def __init__(self, choices, rewards, window=10):
        super().__init__(choices, rewards)

        self.window = window

    def win_stay(self):
        """
        Calculates the probability of staying given a win

            P(stay|win) = P(stay,win) / P(win)

        Returns:
            p_stay_given_win : float
                Probability of staying given a win
        """
        return cond_prob(self.stay_signal, self.win_signal)

    def win_switch(self):
        """
        Calculates the probability of switching given a win

            P(switch|win) = P(switch,win) / P(win)

        Returns:
            p_switch_given_win : float
                Probability of switching given a win
        """
        return cond_prob(self.switch_signal, self.win_signal)

    def lose_stay(self):
        """
        Calculates the probability of staying given a lose

            P(stay|lose) = P(stay,lose) / P(lose)

        Returns:
            p_stay_given_lose : float
                Probability of staying given a lose
        """
        return cond_prob(self.stay_signal, self.lose_signal)

    def lose_switch(self):
        """
        Calculates the probability of switching given a lose

            P(switch|lose) = P(switch,lose) / P(lose)

        Returns:
            p_switch_given_lose : float
                Probability of switching given a lose
        """
        return cond_prob(self.switch_signal, self.lose_signal)


class EntropyMetrics(ProbabilityMetrics):
    """

    """
    def __init__(self, choices, rewards, epsilon=1e-10):
        super().__init__(choices, rewards)

        self.epsilon = epsilon

    def erds_win(self):
        """
        Calculates the Entropy of Reward-Dependent Strategy (ERDS) for win

            ERDS_{win} = H(str|win) = -P(stay,win) x log(P(stay|win)) - P(switch,win) x log(P(switch|win))

        Returns:
            float: ERDS for win
        """
        win_stay = self.win_stay() + self.epsilon
        win_switch = self.win_switch() + self.epsilon

        erds = -joint_prob(self.stay_signal, self.win_signal) * np.log2(win_stay)
        erds -= joint_prob(self.switch_signal, self.win_signal) * np.log2(win_switch)

        return erds

    def erds_lose(self):
        """
        Calculates the Entropy of Reward-Dependent Strategy (ERDS) for lose

            ERDS_{lose} = H(str|lose) = -P(stay,lose) x log(P(stay|lose)) - P(switch,lose) x log(P(switch|lose))

        Returns:
            float: ERDS for lose
        """
        lose_stay = self.lose_stay() + self.epsilon
        lose_switch = self.lose_switch() + self.epsilon

        erds = -joint_prob(self.stay_signal, self.lose_signal) * np.log2(lose_stay)
        erds -= joint_prob(self.switch_signal, self.lose_signal) * np.log2(lose_switch)

        return erds

    def erds_total(self):
        """
        Calculates the total Entropy of Reward-Dependent Strategy (ERDS)

            ERDS = ERDS_{win} + ERDS_{lose} = H(str|reward)

        Returns:
            float: Total ERDS
        """
        return self.erds_win() + self.erds_lose()

    def eods_plus(self, option):
        """
        Calculates the Entropy of Option-Dependent Strategy (EODS). Defined as:

            EODS_{+} = H(str|option=1)
                     = -P(stay,option=1) x log(P(stay|option=1)) - P(switch,option=1) x log(P(switch|option=1))

        Args:
            option (list or np.array): option signal

        Returns:
            float: EODS
        """
        option_pos = win_signal(option)
        pos_stay = cond_prob(self.stay_signal, option_pos) + self.epsilon
        pos_switch = cond_prob(self.switch_signal, option_pos) + self.epsilon

        eods = -joint_prob(self.stay_signal, option_pos) * np.log2(pos_stay)
        eods -= joint_prob(self.switch_signal, option_pos) * np.log2(pos_switch)

        return eods

    def eods_minus(self, option):
        """
        Calculates the Entropy of Option-Dependent Strategy (EODS). Defined as:

            EODS_{-} = H(str|option=0)
                     = -P(stay,option=0) x log(P(stay|option=0)) - P(switch,option=0) x log(P(switch|option=0))

        Args:
            option (list or np.array): option signal

        Returns:
            float: EODS
        """
        option_neg = lose_signal(option)
        neg_stay = cond_prob(self.stay_signal, option_neg) + self.epsilon
        neg_switch = cond_prob(self.switch_signal, option_neg) + self.epsilon

        eods = -joint_prob(self.stay_signal, option_neg) * np.log2(neg_stay)
        eods -= joint_prob(self.switch_signal, option_neg) * np.log2(neg_switch)

        return eods

    def eods_total(self, option):
        """
        Calculates the total Entropy of Option-Dependent Strategy (EODS)

            EODS = EODS_{+} + EODS_{-}

        Args:
            option (list or np.array): option signal

        Returns:
            float: Total EODS
        """
        return self.eods_plus(option) + self.eods_minus(option)

    def erods(self, option):
        """
        Calculates the Entropy of Reward-Option Dependent Strategy (ERODS). Defined as:

            ERODS = H(str|rew,option) = -P(stay,win,opt_pos) x log(P(stay,win,opt_pos)/P(win,opt_pos))
                                        -P(stay,win,opt_neg) x log(P(stay,win,opt_neg)/P(win,opt_neg))
                                        -P(switch,win,opt_pos) x log(P(switch,win,opt_pos)/P(win,opt_pos))
                                        -P(switch,win,opt_neg) x log(P(switch,win,opt_neg)/P(win,opt_neg))
                                        -P(stay,lose,opt_pos) x log(P(stay,lose,opt_pos)/P(lose,opt_pos))
                                        -P(stay,lose,opt_neg) x log(P(stay,lose,opt_neg)/P(lose,opt_neg))
                                        -P(switch,lose,opt_pos) x log(P(switch,lose,opt_pos)/P(lose,opt_pos))
                                        -P(switch,lose,opt_neg) x log(P(switch,lose,opt_neg)/P(lose,opt_neg))

        Args:
            option (list or np.array): option signal

        Returns:
            float: ERODS
        """
        option_pos = win_signal(option)
        option_neg = lose_signal(option)

        stay_win_pos = three_way_prob(self.stay_signal, self.win_signal, option_pos)
        stay_win_neg = three_way_prob(self.stay_signal, self.win_signal, option_neg)
        switch_win_pos = three_way_prob(self.switch_signal, self.win_signal, option_pos)
        switch_win_neg = three_way_prob(self.switch_signal, self.win_signal, option_neg)
        stay_lose_pos = three_way_prob(self.stay_signal, self.lose_signal, option_pos)
        stay_lose_neg = three_way_prob(self.stay_signal, self.lose_signal, option_neg)
        switch_lose_pos = three_way_prob(self.switch_signal, self.lose_signal, option_pos)
        switch_lose_neg = three_way_prob(self.switch_signal, self.lose_signal, option_neg)

        win_pos = joint_prob(self.win_signal, option_pos)
        win_neg = joint_prob(self.win_signal, option_neg)
        lose_pos = joint_prob(self.lose_signal, option_pos)
        lose_neg = joint_prob(self.lose_signal, option_neg)

        erods = -stay_win_pos * np.log2(stay_win_pos / win_pos)
        erods -= stay_win_neg * np.log2(stay_win_neg / win_neg)
        erods -= switch_win_pos * np.log2(switch_win_pos / win_pos)
        erods -= switch_win_neg * np.log2(switch_win_neg / win_neg)
        erods -= stay_lose_pos * np.log2(stay_lose_pos / lose_pos)
        erods -= stay_lose_neg * np.log2(stay_lose_neg / lose_neg)
        erods -= switch_lose_pos * np.log2(switch_lose_pos / lose_pos)
        erods -= switch_lose_neg * np.log2(switch_lose_neg / lose_neg)

        return erods
