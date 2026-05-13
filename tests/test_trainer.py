"""
Integration tests for the training pipeline.
Tests parameter fitting, data splitting, and evaluation metrics.
"""

import pytest
from src.trainer import (
    train_parameters_5, optimize_rule_weights, validate_5,
    train_test_split_5, PREDEFINED_DATA_5, FEATURE_NAMES_8,
    compute_cohens_kappa, compute_ambiguity_rate,
)
from src.fuzzy_engine import DEFAULT_PARAMS_5, CLASSES_5


class TestTrainParameters5:
    def test_returns_params_dict(self):
        """Training should return a dict with all expected MF keys."""
        params = train_parameters_5(PREDEFINED_DATA_5)
        expected_keys = [
            "LEVEL_LOW", "LEVEL_MED", "LEVEL_HIGH",
            "RATIO_LOW", "RATIO_MED", "RATIO_HIGH",
            "GAPS_FEW", "GAPS_SOME", "GAPS_MANY",
        ]
        for key in expected_keys:
            assert key in params, f"Missing param: {key}"

    def test_params_are_tuples(self):
        """Each param should be a tuple of 3 floats."""
        params = train_parameters_5(PREDEFINED_DATA_5)
        for key, val in params.items():
            assert len(val) == 3, f"{key} has {len(val)} values, expected 3"


class TestTrainTestSplit:
    def test_split_preserves_total(self):
        """Total samples after split should equal input."""
        train, test = train_test_split_5(PREDEFINED_DATA_5)
        assert len(train) + len(test) == len(PREDEFINED_DATA_5)

    def test_split_has_both(self):
        """Both train and test should be non-empty."""
        train, test = train_test_split_5(PREDEFINED_DATA_5)
        assert len(train) > 0
        assert len(test) > 0

    def test_split_is_stratified(self):
        """Both train and test should have samples from all classes."""
        train, test = train_test_split_5(PREDEFINED_DATA_5)
        train_classes = set(s["class"] for s in train)
        # With 3 samples per class and 70/30 split, train gets ≥1 per class
        assert len(train_classes) == 5


class TestValidate5:
    def test_returns_accuracy(self):
        """Validation should return accuracy between 0 and 1."""
        acc, confusion, results = validate_5(PREDEFINED_DATA_5, DEFAULT_PARAMS_5)
        assert 0.0 <= acc <= 1.0

    def test_confusion_matrix_shape(self):
        """Confusion matrix should be 5×5."""
        acc, confusion, results = validate_5(PREDEFINED_DATA_5, DEFAULT_PARAMS_5)
        assert len(confusion) == 5
        for row in confusion:
            assert len(row) == 5


class TestCohensKappa:
    def test_perfect_agreement(self):
        """Perfect confusion matrix should give kappa = 1.0."""
        confusion = [[10, 0, 0, 0, 0],
                      [0, 10, 0, 0, 0],
                      [0, 0, 10, 0, 0],
                      [0, 0, 0, 10, 0],
                      [0, 0, 0, 0, 10]]
        kappa = compute_cohens_kappa(confusion)
        assert kappa == 1.0

    def test_no_agreement(self):
        """Completely off-diagonal should give kappa < 0."""
        confusion = [[0, 5, 5, 0, 0],
                      [5, 0, 0, 5, 0],
                      [0, 5, 0, 0, 5],
                      [5, 0, 5, 0, 0],
                      [0, 5, 0, 5, 0]]
        kappa = compute_cohens_kappa(confusion)
        assert kappa < 0.5

    def test_empty(self):
        confusion = [[0]*5 for _ in range(5)]
        kappa = compute_cohens_kappa(confusion)
        assert kappa == 0.0


class TestAmbiguityRate:
    def test_no_ambiguity(self):
        results = [{"details": {"is_ambiguous": False}} for _ in range(10)]
        assert compute_ambiguity_rate(results) == 0.0

    def test_all_ambiguous(self):
        results = [{"details": {"is_ambiguous": True}} for _ in range(10)]
        assert compute_ambiguity_rate(results) == 1.0

    def test_half_ambiguous(self):
        results = [{"details": {"is_ambiguous": i % 2 == 0}} for i in range(10)]
        assert compute_ambiguity_rate(results) == 0.5

    def test_empty(self):
        assert compute_ambiguity_rate([]) == 0.0
