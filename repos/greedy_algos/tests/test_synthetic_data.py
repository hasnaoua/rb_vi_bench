"""Reproducibility of the synthetic Gaussian dataset."""
from __future__ import annotations

import numpy as np

from greedy.synthetic_data.gaussian_data import create_data, gaussian_basis


def basis() -> np.ndarray:
    _, functions = gaussian_basis(dim_basis=2, width=0.08, centers=[0.4, 0.8])
    return functions


def test_same_seed_reproduces_the_same_snapshots():
    """
    The property the synthetic experiments rest on. create_data used to draw
    from the unseeded global np.random, so results/synthetic was regenerated
    differently on every run and none of its numbers could be re-checked --
    while pipelines.synthetic's rng_seed made it look reproducible by only
    seeding the train/test shuffle.
    """
    first = create_data(basis(), num_fields=25, rng=123)
    second = create_data(basis(), num_fields=25, rng=123)

    np.testing.assert_array_equal(first, second)


def test_different_seeds_give_different_snapshots():
    assert not np.array_equal(
        create_data(basis(), num_fields=25, rng=123),
        create_data(basis(), num_fields=25, rng=124),
    )


def test_a_generator_instance_is_accepted():
    np.testing.assert_array_equal(
        create_data(basis(), num_fields=10, rng=np.random.default_rng(7)),
        create_data(basis(), num_fields=10, rng=np.random.default_rng(7)),
    )


def test_noise_is_drawn_from_the_seeded_stream_too():
    noisy = create_data(basis(), num_fields=10, noise_level=0.01, rng=5)
    same = create_data(basis(), num_fields=10, noise_level=0.01, rng=5)
    np.testing.assert_array_equal(noisy, same)


def test_data_stays_in_the_cone():
    """Snapshots feed cone greedies, so they must be nonnegative."""
    assert create_data(basis(), num_fields=30, noise_level=0.05, rng=1).min() >= 0.0


def test_omitting_the_seed_stays_random():
    """The old unseeded behaviour remains available for callers that want it."""
    assert not np.array_equal(
        create_data(basis(), num_fields=25),
        create_data(basis(), num_fields=25),
    )
