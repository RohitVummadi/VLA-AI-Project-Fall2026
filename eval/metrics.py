"""Planar ego-frame ADE/FDE and paired bootstrap confidence intervals.

Run examples: python eval/metrics.py
Import: from eval.metrics import trajectory_metrics, paired_bootstrap_ci

Inputs must already share the ego frame at t0, use meters, and sample
t0 + 0.1 through t0 + 6.4 seconds. This module does not transform frames,
align clip IDs, select model candidates, or run model inference.
"""

import numpy as np


HORIZON_STEPS = {"1s": 10, "3s": 30, "6.4s": 64}


def trajectory_metrics(prediction, ground_truth):
    """Return per-clip x-y ADE/FDE arrays at 1, 3, and 6.4 seconds.

    Both inputs must have matching shape (N, 64, 2) or (N, 64, 3).
    N must be positive. For 3D inputs, height is not evaluated.
    ADE averages all sampled errors through the horizon; FDE uses its
    endpoint. The t0 position is excluded. Results are in meters.
    Retain per-clip values for paired comparisons; use their mean for
    dataset-level scores.
    """
    prediction = np.asarray(prediction, dtype=np.float64)
    ground_truth = np.asarray(ground_truth, dtype=np.float64)

    if prediction.shape != ground_truth.shape:
        raise ValueError("Prediction and ground truth shapes must match")
    if (
        prediction.ndim != 3
        or prediction.shape[0] == 0
        or prediction.shape[1] != 64
        or prediction.shape[2] not in (2, 3)
    ):
        raise ValueError("Expected shape (N, 64, 2) or (N, 64, 3)")
    if not (
        np.isfinite(prediction).all()
        and np.isfinite(ground_truth).all()
    ):
        raise ValueError("Trajectories must contain only finite values")

    errors = np.linalg.norm(
        prediction[..., :2] - ground_truth[..., :2], axis=-1
    )
    results = {}
    for horizon, steps in HORIZON_STEPS.items():
        results[f"ade_{horizon}"] = errors[:, :steps].mean(axis=1)
        results[f"fde_{horizon}"] = errors[:, steps - 1]
    return results


def paired_bootstrap_ci(
    scores_a, scores_b, n_resamples=1000, confidence=0.95, seed=42
):
    """Percentile bootstrap CI for the mean paired difference A minus B.

    Supply matching 1D arrays containing one score per clip, aligned by
    clip ID in identical order. Resampling is over clip pairs, not time
    points. A negative difference favors A for error metrics. Defaults:
    1,000 resamples, a 95% interval, and random seed 42.

    Assumes clips are independent resampling units; correlated clips
    from the same drive may require a drive-level bootstrap instead.
    """
    scores_a = np.asarray(scores_a, dtype=np.float64)
    scores_b = np.asarray(scores_b, dtype=np.float64)
    if (
        scores_a.ndim != 1
        or scores_b.ndim != 1
        or scores_a.shape != scores_b.shape
        or scores_a.size < 2
    ):
        raise ValueError("Provide matching 1D arrays with at least two pairs")
    if not (np.isfinite(scores_a).all() and np.isfinite(scores_b).all()):
        raise ValueError("Scores must contain only finite values")
    if (
        isinstance(n_resamples, bool)
        or not isinstance(n_resamples, (int, np.integer))
        or n_resamples < 1
    ):
        raise ValueError("n_resamples must be a positive integer")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")

    differences = scores_a - scores_b
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(n_resamples)
    for i in range(n_resamples):
        indices = rng.integers(0, len(differences), size=len(differences))
        bootstrap_means[i] = differences[indices].mean()

    alpha = 1 - confidence
    lower, upper = np.quantile(
        bootstrap_means, [alpha / 2, 1 - alpha / 2]
    )
    return {
        "mean_difference": float(differences.mean()),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
    }


def main():
    """Run illustrative sanity checks, not actual model evaluation."""
    truth = np.zeros((3, 64, 3))
    prediction = truth.copy()
    prediction[..., 0] = 3
    prediction[..., 1] = 4

    print("Constant 3-4 offset: every metric should be 5 meters.")
    scores = trajectory_metrics(prediction, truth)
    for name, values in scores.items():
        np.testing.assert_allclose(values, 5.0)
        print(f"{name}: per-clip={values}, mean={values.mean():.3f} m")

    print("\nIncreasing error: verifies horizon indexing.")
    ramp = np.zeros((1, 64, 2))
    ramp[0, :, 0] = np.arange(1, 65) * 0.1
    ramp_scores = trajectory_metrics(ramp, np.zeros_like(ramp))
    expected = {
        "ade_1s": 0.55, "fde_1s": 1.0,
        "ade_3s": 1.55, "fde_3s": 3.0,
        "ade_6.4s": 3.25, "fde_6.4s": 6.4,
    }
    for name, value in expected.items():
        np.testing.assert_allclose(ramp_scores[name], value)
        print(f"{name}: {ramp_scores[name][0]:.3f} m")

    print("\nIllustrative paired comparison: all differences are -0.5 m.")
    method_a = np.array([1.0, 2.0, 1.5, 2.5])
    method_b = np.array([1.5, 2.5, 2.0, 3.0])
    result = paired_bootstrap_ci(method_a, method_b)
    np.testing.assert_allclose(list(result.values()), -0.5)
    print(result)
    print("\nSanity checks passed. Run: python -m pytest -v eval/test_metrics.py")


if __name__ == "__main__":
    main()
