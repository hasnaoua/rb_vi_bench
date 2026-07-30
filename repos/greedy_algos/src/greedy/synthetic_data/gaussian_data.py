import numpy as np
from typing import List, Optional, Tuple


def gaussian_fct(x: np.ndarray, center: float, width: float = 0.08) -> np.ndarray:
    """
    Compute a Gaussian bump centered at `center`.

    Parameters
    ----------
    x : np.ndarray
        Input grid points.
    center : float
        Center of the Gaussian.
    width : float, optional
        Standard deviation of the Gaussian (must be > 0).

    Returns
    -------
    np.ndarray
        Gaussian evaluated at x.
    """
    if width <= 0:
        raise ValueError("width must be > 0")

    return np.exp(-((x - center) ** 2) / (2 * width**2))


def gaussian_basis(
    dim_basis: int = 1,
    width: float = 0.08,
    discretization_count: int = 500,
    centers: Optional[List[float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Gaussian basis functions over the interval [0, 1].

    Parameters
    ----------
    dim_basis : int, optional
        Number of Gaussian basis functions.
    width : float, optional
        Standard deviation of each Gaussian.
    discretization_count : int, optional
        Number of spatial discretization points.
    centers : list of float, optional
        Centers of the Gaussians. If None, centers are evenly spaced.

    Returns
    -------
    x : np.ndarray
        Discretized spatial grid.
    basis : np.ndarray
        Basis functions of shape (dim_basis, discretization_count).
    """
    if dim_basis < 1:
        raise ValueError("dim_basis must be >= 1")

    x = np.linspace(0.0, 1.0, discretization_count)

    if dim_basis == 1:
        return x, np.array([gaussian_fct(x, center=0.5, width=width)])

    if centers is None:
        centers = np.linspace(0.0, 1.0, dim_basis)

    basis = np.array([gaussian_fct(x, c, width) for c in centers])
    return x, basis


def create_data(
    basis_functions: np.ndarray,
    num_fields: int = 100,
    noise_level: float = 0.0,
    min_value: float = 0.0,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """
    Generate synthetic data by random linear combinations of basis functions.

    Parameters
    ----------
    basis_functions : np.ndarray
        Basis functions of shape (num_basis, num_points).
    num_fields : int, optional
        Number of synthetic samples to generate.
    noise_level : float, optional
        Standard deviation of added Gaussian noise.
    min_value : float, optional
        Minimum value for clipping.
    rng : np.random.Generator or int, optional
        Generator (or seed) for the coefficients and noise. Pass one to make the
        dataset reproducible.

        This used to draw from the unseeded global ``np.random``, which silently
        made every caller irreproducible: ``pipelines.synthetic`` takes an
        ``rng_seed`` and seeds its own generator, but that only ever controlled
        the train/test shuffle -- the snapshots themselves were freshly random on
        every run, so no synthetic result could be reproduced or re-checked.
        ``None`` keeps the old unseeded behaviour for any caller that wants it.

    Returns
    -------
    np.ndarray
        Synthetic data of shape (num_fields, num_points).
    """
    num_basis, _ = basis_functions.shape
    generator = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng

    coefficients = generator.uniform(size=(num_fields, num_basis))
    data = coefficients @ basis_functions

    if noise_level > 0:
        data += generator.normal(scale=noise_level, size=data.shape)

    np.clip(data, min_value, None, out=data)
    return data
