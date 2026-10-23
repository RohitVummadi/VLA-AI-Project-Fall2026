"""Hand-made tests for planar ADE/FDE and paired bootstrap intervals.

From the project root: python -m pytest -v eval/test_metrics.py
"""

import numpy as np
import pytest

from eval.metrics import trajectory_metrics, paired_bootstrap_ci


def test_perfect_prediction():
    """Identical moving trajectories have zero error at every horizon."""
    truth = np.zeros((2, 64, 3))
    truth[:, :, 0] = np.arange(1, 65) * 0.1
    for values in trajectory_metrics(truth.copy(), truth).values():
        np.testing.assert_allclose(values, [0, 0])


def test_constant_three_four_offset():
    """A 3 m x-offset and 4 m y-offset give sqrt(9+16) = 5 m."""
    truth = np.zeros((2, 64, 3))
    prediction = truth.copy()
    prediction[..., 0] = 3
    prediction[..., 1] = 4
    for values in trajectory_metrics(prediction, truth).values():
        np.testing.assert_allclose(values, [5, 5])


def test_increasing_error_at_all_horizons():
    """Errors 0.1,...,6.4 have arithmetic-series means and known endpoints."""
    truth = np.zeros((1, 64, 2))
    prediction = truth.copy()
    prediction[0, :, 0] = np.arange(1, 65) * 0.1
    expected = {
        "ade_1s": 0.55, "fde_1s": 1.0,
        "ade_3s": 1.55, "fde_3s": 3.0,
        "ade_6.4s": 3.25, "fde_6.4s": 6.4,
    }
    scores = trajectory_metrics(prediction, truth)
    assert scores.keys() == expected.keys()
    for name, answer in expected.items():
        np.testing.assert_allclose(scores[name], [answer])


def test_horizon_endpoint_impulses():
    """Errors only at steps 10,30,64 verify endpoint inclusion precisely."""
    truth = np.zeros((1, 64, 2))
    prediction = truth.copy()
    prediction[0, [9, 29, 63], 0] = [10, 30, 64]
    scores = trajectory_metrics(prediction, truth)
    expected = {
        "ade_1s": 10 / 10, "fde_1s": 10,
        "ade_3s": 40 / 30, "fde_3s": 30,
        "ade_6.4s": 104 / 64, "fde_6.4s": 64,
    }
    for name, answer in expected.items():
        np.testing.assert_allclose(scores[name], [answer])


def test_separate_scores_for_each_clip():
    """A perfect clip and a 2 m offset clip remain separate scores."""
    truth = np.zeros((2, 64, 2))
    prediction = truth.copy()
    prediction[1, :, 1] = -2
    for values in trajectory_metrics(prediction, truth).values():
        np.testing.assert_allclose(values, [0, 2])
        assert values.mean() == pytest.approx(1)


def test_height_is_excluded_from_planar_error():
    """A pure 10 m height difference gives zero planar error."""
    truth = np.zeros((1, 64, 3))
    prediction = truth.copy()
    prediction[..., 2] = 10
    for values in trajectory_metrics(prediction, truth).values():
        np.testing.assert_allclose(values, [0])


def test_bootstrap_identical_methods():
    scores = np.array([1.0, 10.0, 100.0])
    result = paired_bootstrap_ci(scores, scores)
    for value in result.values():
        assert value == pytest.approx(0)


def test_bootstrap_preserves_pairs_and_difference_sign():
    """Every paired difference is -0.5 despite large between-clip variation."""
    a = np.array([1.0, 100.0, 1000.0])
    b = a + 0.5
    result = paired_bootstrap_ci(a, b, n_resamples=1000, seed=42)
    assert set(result) == {"mean_difference", "ci_lower", "ci_upper"}
    for value in result.values():
        assert value == pytest.approx(-0.5)
    reverse = paired_bootstrap_ci(b, a)
    for value in reverse.values():
        assert value == pytest.approx(0.5)


def test_variable_bootstrap_matches_exact_small_sample():
    """Differences [0,-2]: possible resampled means are 0,-1,-2.

    The four equally likely ordered resamples give means [0,-1,-1,-2].
    The exact bootstrap distribution has 2.5% and 97.5% quantiles -2,0.
    With the fixed seed and 1,000 resamples, empirical endpoints match.
    """
    result = paired_bootstrap_ci([0, 0], [0, 2], seed=42)
    assert result["mean_difference"] == pytest.approx(-1)
    assert result["ci_lower"] == pytest.approx(-2)
    assert result["ci_upper"] == pytest.approx(0)


def test_bootstrap_reproducible_for_fixed_seed():
    a = [0, 2, 1, 5, 4]
    b = [1, 1, 3, 2, 6]
    assert paired_bootstrap_ci(a, b, seed=42) == paired_bootstrap_ci(
        a, b, seed=42
    )


@pytest.mark.parametrize("shape", [(64, 2), (1, 63, 2), (1, 64, 4), (0, 64, 2)])
def test_reject_invalid_trajectory_shape(shape):
    values = np.zeros(shape)
    with pytest.raises(ValueError):
        trajectory_metrics(values, values)


def test_reject_mismatched_trajectory_shapes():
    with pytest.raises(ValueError):
        trajectory_metrics(np.zeros((1, 64, 2)), np.zeros((2, 64, 2)))


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_reject_nonfinite_trajectory(bad_value):
    truth = np.zeros((1, 64, 2))
    prediction = truth.copy()
    prediction[0, 0, 0] = bad_value
    with pytest.raises(ValueError):
        trajectory_metrics(prediction, truth)


@pytest.mark.parametrize("a,b", [([], []), ([1], [1]), ([1, 2], [1]), ([[1, 2]], [[1, 2]]), ([1, np.nan], [1, 2]), ([1, 2], [1, np.inf])])
def test_reject_invalid_bootstrap_scores(a, b):
    with pytest.raises(ValueError):
        paired_bootstrap_ci(a, b)


@pytest.mark.parametrize("confidence", [0, 1, -0.1, 1.1])
def test_reject_invalid_confidence(confidence):
    with pytest.raises(ValueError):
        paired_bootstrap_ci([1, 2], [2, 3], confidence=confidence)


@pytest.mark.parametrize("n_resamples", [0, -1, 1.5, True])
def test_reject_invalid_resample_count(n_resamples):
    with pytest.raises(ValueError):
        paired_bootstrap_ci([1, 2], [2, 3], n_resamples=n_resamples)
