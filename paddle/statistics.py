from typing import Any, Optional

import numpy as np


def gstd(a: Any, weights: Optional[Any] = None) -> float:
    """Calculate the (weighted) geometric standard deviation.

    Based on:
        https://www.researchgate.net/post/How-can-I-calculate-the-value-of-the-geometric-standard-deviation-taking-into-account-weight/53721a9ed5a3f2d33c8b4607/citation/download

    :param a: List, Tuple or numpy array of numbers.
    :param weights: Weights associated with the values in `a`.
    :return: (weighted) geometric standard deviation
    """
    # TODO: Replace Any with ArrayLike, as soon as it is included in numpy.

    log_a = np.log(a)

    if weights is not None:
        if len(weights) == 0:
            weights = None

    average = np.average(log_a, weights=weights)
    variance = np.average((log_a - average) ** 2, weights=weights)

    return np.exp(np.sqrt(variance))


# Customization of scipy.stats.gmean
# See also: https://github.com/scipy/scipy/issues/13065
# TODO: Replace with scipy version, when scipy 1.7.0 is released.
def gmean(a, axis=0, dtype=None, weights=None):
    """
    Compute the geometric mean along the specified axis.
    Return the geometric average of the array elements.
    That is:  n-th root of (x1 * x2 * ... * xn)
    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    axis : int or None, optional
        Axis along which the geometric mean is computed. Default is 0.
        If None, compute over the whole array `a`.
    dtype : dtype, optional
        Type of the returned array and of the accumulator in which the
        elements are summed. If dtype is not specified, it defaults to the
        dtype of a, unless a has an integer dtype with a precision less than
        that of the default platform integer. In that case, the default
        platform integer is used.
    weights : array_like, optional
        The weights array can either be 1-D (in which case its length must be
        the size of `a` along the given `axis`) or of the same shape as `a`.
        Default is None, which gives each value a weight of 1.0.
    Returns
    -------
    gmean : ndarray
        See `dtype` parameter above.
    See Also
    --------
    numpy.mean : Arithmetic average
    numpy.average : Weighted average
    hmean : Harmonic mean
    Notes
    -----
    The geometric average is computed over a single dimension of the input
    array, axis=0 by default, or all values in the array if axis=None.
    float64 intermediate and return values are used for integer inputs.
    Use masked arrays to ignore any non-finite values in the input or that
    arise in the calculations such as Not a Number and infinity because masked
    arrays automatically mask any non-finite values.
    References
    ----------
    .. [1] "Weighted Geometric Mean", *Wikipedia*, https://en.wikipedia.org/wiki/Weighted_geometric_mean.
    Examples
    --------
    >>> from scipy.stats import gmean
    >>> gmean([1, 4])
    2.0
    >>> gmean([1, 2, 3, 4, 5, 6, 7])
    3.3800151591412964
    """
    if not isinstance(a, np.ndarray):
        # if not an ndarray object attempt to convert it
        log_a = np.log(np.array(a, dtype=dtype))
    elif dtype:
        # Must change the default dtype allowing array type
        if isinstance(a, np.ma.MaskedArray):
            log_a = np.log(np.ma.asarray(a, dtype=dtype))
        else:
            log_a = np.log(np.asarray(a, dtype=dtype))
    else:
        log_a = np.log(a)

    if weights is not None:
        if len(weights) == 0:
            weights = None
        else:
            weights = np.asanyarray(weights, dtype=dtype)

    return np.exp(np.average(log_a, axis=axis, weights=weights))
