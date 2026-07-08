import numpy as np

from reduction_common import _compute_gain_series


def test_relative_gain_uses_previous_error_normalization():
    errors = np.array([10.0, 7.0, 4.0], dtype=float)

    gains = _compute_gain_series(errors, "relative")

    np.testing.assert_allclose(gains, np.array([0.3, 0.42857142857142855]))
