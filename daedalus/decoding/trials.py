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
from shutil import rmtree
from joblib import Parallel, delayed, Memory

import numpy as np
from sklearn.preprocessing import StandardScaler
from skleanr.impute import KNNImputer
from sklearn.svm import SVC
from sklearn.enemble import HistGradientBoostingClassifier
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

        # Preproess
        scaler = StandardScaler()
        imp = KNNImputer(n_neighbors=self.params["n_neighbors"], add_indicator=True)

        # Classifier
        svm = SVC(class_weight="balanced")

        # Pipeline
        pipe = Pipeline(
            steps=[
                ("scalar", scaler),
                ("imp", imp),
                ("clf", svm),
            ],
            memory=self.memory,
            verbose=self.params["verbose"],
            )

        # Cross validation
        kf = StratifiedKFold(
            n_splits=self.params["cv_splits"],
            shuffle=False,
            )

        # Randomized search
        self.clf = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=self.params["param_dist"],
            n_iter=self.params["n_iter"],
            scoring=self.params["scoring"],
            n_jobs=self.params["n_jobs"],
            refit=True,
            cv=kf,
            verbose=self.params["verbose"]
            )

    def magic(self, X_train, y_train, X_test, y_test):
        """
        Calculate the accuracy and f1 score for a given fold.

        Args:
            X_train (np.array): The training features.
            y_train (np.array): The training labels.

        Returns:
            metrics (tuple): The metrics for the decoding analysis.
        """
        # Time
        self.clock.tick()

        # Fit the training data
        self.clf.fit(X_train, y_train)

        # Get the predicted labels for the test set
        y_pred = self.clf.predict(X_test)

        # Probability of labels
        true_prob = best_clf.predict_proba(X_test)[0, y_test]
        confidence_prob = best_clf.predict_proba(X_test)[0, y_pred]

        metrics = (y_pred[0], true_prob[0], confidence_prob[0])

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

    def compute_roc(self, X: np.ndarray, y: np.ndarray, classifier):
        """
        Compute the ROC curve for the given classifier.

        Args:
            X : ndarray
                The features for the decoding analysis.

            y : ndarray
                The labels for the decoding analysis.

            classifier : sklearn classifier
                The best classifier found during decoding.

        Returns:
            fpr : ndarray
                The false positive rate for the ROC curve.

            tpr : ndarray
                The true positive rate for the ROC curve.

            roc_auc : float
                The area under the ROC curve.
        """
        if self._clf_name == 'SVC':
            # Get the decision function values for the test set
            y_score = classifier.decision_function(X)
        else:
            # Get the probability of predicted labels for the test set
            y_score = classifier.predict_proba(X)[:, 1]

        # Get number of classes
        n_classes = len(np.unique(y))

        # Binary or multiclasss
        if n_classes == 2:

            # Binarize the labels
            binary_y_test = y.astype(float).astype(int)

            # Compute the ROC curve
            fpr, tpr, _ = roc_curve(binary_y_test, y_score)

            # Compute the area under the ROC curve
            roc_auc = roc_auc_score(binary_y_test, y_score)

        elif n_classes > 2:

            # Binarize the labels
            binary_y_test = label_binarize(y, classes=list(range(n_classes)))

            # Compute the area under the ROC curve
            roc_auc = roc_auc_score(binary_y_test, y_score, average='micro', multi_class='ovr')

            # Placeholders for fpr and tpr
            fpr, tpr = 0, 0

        else:

            raise ValueError(
                f"Number of classes must be equal or greater than 2. The number of classes is {n_classes}."
            )

        return fpr, tpr, roc_auc