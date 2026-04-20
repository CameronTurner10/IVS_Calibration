import pytest
import numpy as np

from src.smoothing_spline.implementation.spline_model import (
    build_observation_matrix,
    build_Q_matrix,
    build_R_matrix,
    evaluate_spline,
)

# ./.venv/bin/pytest


@pytest.fixture
def equal_spacing_n4():
    """n=4 knots with equal spacing h=5."""
    return np.array([100.0, 105.0, 110.0, 115.0])


def test_build_Q_matrix_shape(equal_spacing_n4):
    """Q must be (n, n-2) = (4, 2) for n=4 (Fengler eq. 13)."""
    Q = build_Q_matrix(equal_spacing_n4)
    assert Q.shape == (4, 2), f"Expected (4, 2), got {Q.shape}"


def test_build_Q_matrix_values(equal_spacing_n4):
    """Verify exact Q values for n=4, equal spacing h=5 (Fengler eq. 13)."""
    Q = build_Q_matrix(equal_spacing_n4)
    h = 5.0
    expected = np.array([
        [ 1/h, 0.0],
        [-2/h, 1/h],
        [ 1/h, -2/h],
        [ 0.0, 1/h],
    ])
    np.testing.assert_allclose(Q, expected, atol=1e-12)


def test_build_R_matrix_shape(equal_spacing_n4):
    """R must be (n-2, n-2) = (2, 2) for n=4 (Fengler eq. 14)."""
    R = build_R_matrix(equal_spacing_n4)
    assert R.shape == (2, 2)


def test_build_R_matrix_values(equal_spacing_n4):
    """Verify exact R values for n=4, equal spacing h=5 (Fengler eq. 14).
    R[0,0] = (h0+h1)/3 = 10/3
    R[0,1] = h1/6 = 5/6
    """
    R = build_R_matrix(equal_spacing_n4)
    expected = np.array([
        [10/3,  5/6],
        [5/6,  10/3]
    ])
    np.testing.assert_allclose(R, expected, atol=1e-12)


def test_build_observation_matrix_unweighted():
    prices = np.array([100.0, 50.0, 5.0])
    W = build_observation_matrix(prices, fit_mode="unweighted")
    np.testing.assert_allclose(W, np.eye(3), atol=1e-12)


def test_build_observation_matrix_weighted_with_floor():
    prices = np.array([100.0, 0.25, 2.0])
    W = build_observation_matrix(prices, fit_mode="weighted", min_price=1.0)
    expected = np.diag([0.01, 1.0, 0.5])
    np.testing.assert_allclose(W, expected, atol=1e-12)


def test_evaluate_spline_right_extrapolation_uses_negative_curvature_term():
    result = {
        "g": np.array([10.0, 6.0, 3.0]),
        "gamma": np.array([0.3]),
        "strikes": np.array([100.0, 110.0, 120.0]),
    }

    assert evaluate_spline(result, 130.0) == pytest.approx(0.0, abs=1e-12)
