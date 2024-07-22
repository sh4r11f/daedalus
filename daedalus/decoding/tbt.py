#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =================================================================================================== #
#
#
#                        SCRIPT: trials.py
#
#
#               DESCRIPTION: Trial by trial decoder
#
#
#                           RULE: DAYW
#
#
#
#                      CREATOR: Sharif Saleki
#                            TIME: 07-18-2024-7810598105114117
#                          SPACE: Dartmouth College, Hanover, NH
#
# =================================================================================================== #
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from .base import Decoder


class TrialSVC(Decoder):
    def __init__(self, **kwargs):
        super().__init__(name="SVC", **kwargs)

    def fit_params(self, X, y):
        """
        Fit the classifier to the average data to find the best params.

        Args:
            X : ndarray
                The features for the decoding analysis.

            y : ndarray
                The labels for the decoding analysis.
        """
        # Clean memory
        self.clean()

        # Preproess
        scaler = StandardScaler()
        imp = KNNImputer(n_neighbors=self.params["n_neighbors"], add_indicator=True)

        # Classifier
        svm = SVC(
            probability=False,
            kernel=self.params["svm_kernel"],
            class_weight="balanced",
            max_iter=self.params["max_iter"],
            )

        # Pipeline
        pipe = Pipeline(
            steps=[
                ("scaler", scaler),
                ("imp", imp),
                ("svm", svm),
            ],
            # memory=self.memory,
            verbose=self.params["verbose"],
            )

        # Cross validation
        kf = StratifiedKFold(
            n_splits=2,
            # n_splits=self.params["cv_splits"],
            shuffle=False,
            )

        # Randomized search
        clf = RandomizedSearchCV(
            estimator=pipe,
            param_distributions={
                # "svm__C": np.logspace(-10, 10, 10), "svm__gamma": np.logspace(-10, 10, 10)
                "svm__C": np.logspace(-10, 10, 2), "svm__gamma": np.logspace(-10, 10, 2)
                },
            n_iter=self.params["n_iter"],
            scoring=self.params["scoring"],
            # n_jobs=1,
            n_jobs=self.params["n_jobs"],
            cv=kf,
            refit=False,
            verbose=self.params["verbose"]
            )

        # Time
        self.clock.tick()

        # Fit the classifier
        clf.fit(X[:20, :], y[:20])

        # Get score and params
        t = self.clock.tock()
        cv_results = clf.cv_results_
        best = {"time": t}
        top_score = np.inf
        for scorer in self.params["scoring"]:
            best_index = cv_results[f'rank_test_{scorer}'].argmin()
            best_score = cv_results[f'mean_test_{scorer}'][best_index]
            if best_score < top_score:
                top_score = best_score
            best_params = {key.replace('param_', ''): cv_results[key][best_index] for key in cv_results if key.startswith('param_')}
            best[scorer] = {
                'score': best_score,
                'params': best_params
            }

        return top_score, best, clf

    def fit_bin(self, X_train, y_train, X_test, y_test, clf_params):
        """
        Calculate the accuracy and f1 score for a given fold.

        Args:
            X_train (np.array): The training features.
            y_train (np.array): The training labels.

        Returns:
            metrics (tuple): The metrics for the decoding analysis.
        """
        # Classifier
        clf = self.pipe.clone()
        clf.set_params(
            svm__kernel=self.params["svm_kernel"],
            probability=True,
            class_weight="balanced",
            C=clf_params["C"],
            gamma=clf_params["gamma"]
            )

        # Time
        self.clock.tick()

        # Fit the training data
        clf.fit(X_train, y_train)

        # Get the predicted labels for the test set
        y_pred = clf.predict(X_test)

        # Probability of labels
        pred = clf.predict_proba(X_test)
        positive_prob = pred[0, y_test]
        confidence_prob = pred[0, y_pred]

        metrics = (y_pred[0], positive_prob[0], confidence_prob[0])

        return metrics

    def compute_time_metrics(
            self,
            time: int,
    ) -> tuple:
        """
        Calculate the accuracy and f1 score for a given fold.

        Args:
            time: int
                The time bin to use for the decoding analysis.

            splitter: sklearn splitter
                The splitter to use for the decoding analysis.

            neuron_filter: int or None
                The neuron to use for the decoding analysis. None means use all neurons (population decoding).

        Returns:
            metrics: tuple
        """
        # print(f"Time bin {time}/{self._num_timepoints}")

        # Get the features and labels for this time bin
        X = self._normal_data[:, time, :]
        y = self._labels

        if neuron_filter is not None:
            X = X[:, neuron_filter]
            X = X[:, np.newaxis]

        # Run the decoding analysis in parallel
        fold_metrics = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(self._compute_fold_metrics)(
                train_idx=train_idx,
                test_idx=test_idx,
                X_train=X,
                X_test=X,
                y=y
            )
            for train_idx, test_idx in splitter.split(X, y)
        )

        # Unpack the trial results
        predictions, true_probs, confidence_probs = zip(*fold_metrics)
        predictions = np.array(predictions)
        true_probs = np.array(true_probs)
        confidence_probs = np.array(confidence_probs)

        # Calculate accuracy
        f1_scores = f1_score(y, predictions, average='weighted')
        accuracy_scores = balanced_accuracy_score(y, predictions)
        auc_scores = roc_auc_score(y, predictions, average='weighted')

        metrics = (true_probs, confidence_probs, f1_scores, accuracy_scores, auc_scores)
