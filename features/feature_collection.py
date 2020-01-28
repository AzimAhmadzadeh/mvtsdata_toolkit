import scipy as sp
from scipy.signal import argrelextrema
import numpy as np
from itertools import *
import pandas as pd
from typing import Union

'''
    This is a collection of all time series features gathered
    to be derived from the multivariate time series of SWAN data
    benchmark.
    NOTE: Each method name MUST start with the string 'get_'.
'''


# -----------------------------------------------------------
#                   TIME SERIES FEATURES                    #
# -----------------------------------------------------------


def get_min(uni_ts: Union[pd.Series, np.ndarray]) -> np.float64:
    """
    :return: The minimum value of a given univariate time series.
    """
    return np.float64(np.min(uni_ts))


def get_max(uni_ts: Union[pd.Series, np.ndarray]) -> np.float64:
    """
    :return: The maximum value of a given univariate time series.
    """
    return np.float64(np.max(uni_ts))


def get_median(uni_ts: Union[pd.Series, np.ndarray]) -> np.float64:
    """
    :return: The median value of a given univariate time series.
    """
    return np.float64(np.median(uni_ts))


def get_mean(uni_ts: Union[pd.Series, np.ndarray]) -> np.float64:
    """
    :return: The arithmetic mean value of a given univariate time series.
    """
    return np.float64(np.mean(uni_ts))


def get_stddev(uni_ts: Union[pd.Series, np.ndarray]) -> np.float64:
    """
    :return: The standard deviation of a given univariate time series.
    """
    return np.float64(np.std(uni_ts))


def get_var(uni_ts: Union[pd.Series, np.ndarray]) -> np.float64:
    """
    :return: The variance of a given univariate time series.
    """
    return np.float64(np.var(uni_ts))


def get_skewness(uni_ts: Union[pd.Series, np.ndarray]):
    """
    :return: The skewness of a given univariate time series.
    """
    return np.float64(sp.stats.skew(uni_ts))


def get_kurtosis(uni_ts: Union[pd.Series, np.ndarray]):
    """
    :return: The kurtosis of a given univariate time series.
    """
    return np.float64(sp.stats.kurtosis(uni_ts))


def get_no_local_maxima(uni_ts: Union[pd.Series, np.ndarray]):
    """
    :return: The number of local maxima in a given univariate time series.
    """
    return len(argrelextrema(uni_ts.values, np.greater)[0])


def get_no_local_minima(uni_ts: Union[pd.Series, np.ndarray]):
    """
    :return: The number of local minima in a given univariate time series.
    """
    return len(argrelextrema(uni_ts.values, np.less)[0])


def get_no_local_extrema(uni_ts: Union[pd.Series, np.ndarray]):
    """
    :return: The number of local extrema in a given univariate time series.
    """
    return get_no_local_minima(uni_ts) + get_no_local_maxima(uni_ts)


def get_no_zero_crossings(uni_ts: Union[pd.Series, np.ndarray]):
    """
    :return: The number of zero-crossings in a given univariate time series.
    """
    zero_crossings = np.where(np.diff(np.sign(uni_ts)))[0]
    return len(zero_crossings)


def get_mean_local_maxima_value(uni_ts: Union[pd.Series, np.ndarray],
                                only_positive: bool = False) -> np.float64:
    """
    Returns the mean of local maxima values.

    :param uni_ts: Univariate time series.
    :param only_positive: Only positive flag for local maxima. When True only positive local
           maxima are considered. Default is False.

    :return: Mean of local maxima values.
    """
    local_maxima = argrelextrema(uni_ts.values, np.greater)[0]
    maxima_values = uni_ts[local_maxima]
    if only_positive:
        pos_maxima_values = maxima_values[maxima_values > 0]
        return np.float64(np.mean(pos_maxima_values))
    else:
        return np.float64(np.mean(maxima_values))


def get_mean_local_minima_value(uni_ts: Union[pd.Series, np.ndarray], only_negative: bool = False) \
        -> np.float64:
    """
    Returns the mean of local minima values.

    :param uni_ts: Univariate time series.
    :param only_negative: Only negative flag for local minima. When True only negative local
           minima are considered. Default is False.

    :return: Mean of local minima values.
    """
    local_minima = argrelextrema(uni_ts.values, np.less)[0]
    minima_values = uni_ts[local_minima]
    if only_negative:
        neg_minima_values = minima_values[minima_values < 0]
        return np.float64(np.mean(neg_minima_values))
    else:
        return np.float64(np.mean(minima_values))


def get_no_mean_local_maxima_upsurges(uni_ts: Union[pd.Series, np.ndarray],
                                      only_positive: bool = False) -> np.int64:
    """
    Returns the number of values in a given time series whose value is greater than the mean of
    local maxima values (# of upsurges).

    :param uni_ts: Univariate time series.
    :param only_positive: Only positive flag for mean local maxima. When True only positive local
           maxima are considered. Default is False.

    :return: Number of points whose value is greater than mean local maxima.
    """
    mean_local_maxima = get_mean_local_maxima_value(uni_ts, only_positive)
    upsurging = uni_ts > mean_local_maxima
    return np.int64(np.sum(upsurging))


def get_no_mean_local_minima_downslides(uni_ts: Union[pd.Series, np.ndarray],
                                        only_negative: bool = False) -> np.int64:
    """
    Returns the number of values in a given time series whose value is less than the mean of
    local minima values (# of downslides).

    :param uni_ts: Univariate time series.
    :param only_negative: Only negative flag for mean local minima. When True only negative local
           minima are considered. Default is False.

    :return: Number of points whose value is less than mean local minima.
    """
    mean_local_minima = get_mean_local_minima_value(uni_ts, only_negative)
    downslides = uni_ts < mean_local_minima
    return np.int64(np.sum(downslides))


def get_difference_of_mins(uni_ts: Union[pd.Series, np.ndarray]) -> np.float64:
    """
    :return: The absolute difference between the minimums of the first and the second halves of a
             given univariate time series.
    """
    mid = int(len(uni_ts) / 2)
    return np.abs(get_min(uni_ts[:mid]) - get_min(uni_ts[mid:]))


def get_difference_of_maxs(uni_ts: Union[pd.Series, np.ndarray]) -> np.float64:
    """
    :return: The absolute difference between the maximums of the first and the second halves of a
             given univariate time series.
    """
    mid = int(len(uni_ts) / 2)
    return np.abs(get_max(uni_ts[:mid]) - get_max(uni_ts[mid:]))


def get_difference_of_means(uni_ts: Union[pd.Series, np.ndarray]) -> np.float64:
    """
    :return: The absolute difference between the means of the first and the second halves of a
             given univariate time series.
    """
    mid = int(len(uni_ts) / 2)
    return np.abs(get_mean(uni_ts[:mid]) - get_mean(uni_ts[mid:]))


def get_difference_of_stds(uni_ts: Union[pd.Series, np.ndarray]) -> np.float64:
    """
    :return: The absolute difference between the standard dev. of the first and the second halves
             of a given univariate time series.
    """
    mid = int(len(uni_ts) / 2)
    return np.abs(get_stddev(uni_ts[:mid]) - get_stddev(uni_ts[mid:]))


def get_difference_of_vars(uni_ts: Union[pd.Series, np.ndarray]) -> np.float64:
    """
    :return: The absolute difference between the variances of the first and the second halves of
             a given univariate time series.
    """
    mid = int(len(uni_ts) / 2)
    return np.abs(get_var(uni_ts[:mid]) - get_var(uni_ts[mid:]))


def get_difference_of_medians(uni_ts: Union[pd.Series, np.ndarray]) -> np.float64:
    """
    :return: The absolute difference between the medians of the first and the second halves of a
             given univariate time series."""
    mid = int(len(uni_ts) / 2)
    return np.abs(get_median(uni_ts[:mid]) - get_median(uni_ts[mid:]))


def get_dderivative_mean(uni_ts: Union[pd.Series, np.ndarray], step_size: int = 1) -> np.float64:
    """
    :return: The mean of the difference-derivative of univariate time series within the function
             we use step_size to find derivative (default value of step_size is 1).
    """
    return np.float64(get_mean(_difference_derivative(uni_ts, step_size)))


def get_gderivative_mean(uni_ts: Union[pd.Series, np.ndarray]) -> np.float64:
    """
    :return: The mean of the gradient-derivative of univariate time series.
    """
    return get_mean(_gradient_derivative(uni_ts))


def get_dderivative_stddev(uni_ts: Union[pd.Series, np.ndarray], step_size: int = 1) -> np.float64:
    """
    :return: The std.dev of the difference derivative of univariate time series within the
             function we use step_size to find derivative (default value of step_size is 1).
    """
    return get_stddev(_difference_derivative(uni_ts, step_size))


def get_gderivative_stddev(uni_ts: Union[pd.Series, np.ndarray]) -> np.float64:
    """
    :return: The std.dev of the gradient derivative of univariate time series.
    """
    return get_stddev(_gradient_derivative(uni_ts))


def get_dderivative_skewness(uni_ts: Union[pd.Series, np.ndarray], step_size: int = 1) -> np.float64:
    """
    :return: The skewness of the difference derivative of univariate time series within the
             function we use step_size to find derivative (default value of step_size is 1).
    """
    return get_skewness(_difference_derivative(uni_ts, step_size))


def get_gderivative_skewness(uni_ts: Union[pd.Series, np.ndarray]) -> np.float64:
    """
    :return: The skewness of the gradient derivative of the univariate time series.
    """
    return get_skewness(_gradient_derivative(uni_ts))


def get_dderivative_kurtosis(uni_ts: Union[pd.Series, np.ndarray], step_size: int = 1) -> np.float64:
    """
    :return: The kurtosis of the difference derivative of univariate time series within the
             function we use step_size to find derivative (default value of step_size is 1)."""
    return get_kurtosis(_difference_derivative(uni_ts, step_size))


def get_gderivative_kurtosis(uni_ts: Union[pd.Series, np.ndarray]) -> np.float64:
    """
    :return: The kurtosis of the gradient derivative of the univariate time series.
    """
    return get_kurtosis(_gradient_derivative(uni_ts))


def get_linear_weighted_average(uni_ts: Union[pd.Series, np.ndarray]) -> np.float64:
    """
     Computes the linear weighted average of a univariate time series. It simply, for each `x_i` in
     `uni_ts` computes the following::

        2/(n*(n+1)) * sum(i* x_i)

    where `n` is the length of the time series.

    :return: The linear weighted average of `uni_ts`.
    """
    n = len(uni_ts)
    i_val = np.arange(1, n + 1)
    w_sum = np.sum(i_val * uni_ts)
    return (2.0 * w_sum) / (n * (n + 1))


def get_quadratic_weighted_average(uni_ts: Union[pd.Series, np.ndarray]) -> np.float64:
    """
    Computes the quadratic weighted average of a univariate time series. It simply, for each `x_i` in
    `uni_ts`, computes the following::

        6/(n*(n+1)(2*n+1)) * sum(i^2 * x_i)

    where `n` is the length of the time seires.

    :return: The quadratic weighted average of `uni_ts`.
    """
    n = len(uni_ts)
    i_val = np.arange(1, n + 1)
    qw_sum = np.sum(i_val * i_val * uni_ts)
    return (6.0 * qw_sum) / (n * (n + 1) * (2 * n + 1))


def get_average_absolute_change(uni_ts: Union[pd.Series, np.ndarray]) -> np.float64:
    """
    :return: The average absolute first difference of a univariate time series.
    """
    return np.float64(np.mean(np.abs(_difference_derivative(uni_ts))))


def get_average_absolute_derivative_change(uni_ts: Union[pd.Series, np.ndarray]) -> np.float64:
    """
    :return: The average absolute first difference of a derivative of univariate time series.
    """
    return np.float64(np.mean(np.abs(np.diff(uni_ts, 2))))


def get_positive_fraction(uni_ts: Union[pd.Series, np.ndarray]) -> np.float64:
    """
    :return: The fraction of positive numbers in uni_ts.
    """
    return np.float64(np.mean(uni_ts > 0))


def get_negative_fraction(uni_ts: Union[pd.Series, np.ndarray]) -> np.float64:
    """
    :return: The fraction of negative numbers in uni_ts.
    """
    return np.float64(np.mean(uni_ts < 0))


def get_last_K(uni_ts: Union[pd.Series, np.ndarray], k: int) -> pd.Series:
    """
    :return: The last k values in a univariate time series.
    """
    return uni_ts[-k:]


def get_last_value(uni_ts) -> np.float64:
    """
    :return: The last value in a univariate time series. This seems redundant since `get_last_K`
             already does this job, but it is necessary because the return type is different (
             `numpy.int64`) than what `get_last_K` returns (`numpy.ndarray`). This is especially
             important if the methods in this module are going to be called from a list.
    """
    return np.float64(uni_ts.iloc[-1])


def get_sum_of_last_K(uni_ts: Union[pd.Series, np.ndarray], k: int = 10) -> np.float64:
    """
    :return: The sum of last k-values in a univariate time series.
    """
    return np.float64(np.sum(uni_ts[-k:]))


def get_mean_last_K(uni_ts: Union[pd.Series, np.ndarray], k: int = 10) -> np.float64:
    """
    :return: The mean of last k-values in a univariate time series.
    """
    return np.float64(np.mean(uni_ts[-k:]))


def get_longest_positive_run(uni_ts: Union[pd.Series, np.ndarray]) -> np.int64:
    """
    :return: The longest positive run in a univariate time series.
    """
    ts_encode = [(len(list(group)), name) for name, group in groupby(_sign(uni_ts))]
    signs = np.array([signature for run, signature in ts_encode])
    runs = np.array([run for run, signature in ts_encode])
    if runs[signs == 1].size == 0:  # empty array
        return np.int64(0)
    return np.int64(np.max(runs[signs == 1]))


def get_longest_negative_run(uni_ts: Union[pd.Series, np.ndarray]) -> np.int64:
    """
    :return: The longest negative run in a univariate time series.
    """
    ts_encode = [(len(list(group)), name) for name, group in groupby(_sign(uni_ts))]
    signs = np.array([signature for run, signature in ts_encode])
    runs = np.array([run for run, signature in ts_encode])
    if runs[signs == -1].size == 0:  # empty array
        return np.int64(0)
    return np.int64(np.max(runs[signs == -1]))


def get_longest_monotonic_increase(uni_ts: Union[pd.Series, np.ndarray]) -> np.int64:
    """
    :return: The length of the time series segment with the longest monotonic increase.
    """
    return get_longest_positive_run(np.sign(np.diff(uni_ts)))


def get_longest_monotonic_decrease(uni_ts: Union[pd.Series, np.ndarray]) -> np.int64:
    """
    :return: The length of the time series segment with the longest monotonic increase.
    """
    return get_longest_negative_run(np.sign(np.diff(uni_ts)))


def get_slope_of_longest_mono_increase(uni_ts: Union[pd.Series, np.ndarray]) -> np.float64:
    """
    Identifies the longest monotonic increase and gets the slope.

    :return: The slope of the longest monotonic increase in `uni_ts`.
    """
    ts_encode = [(len(list(group)), name) for name, group in groupby(_sign(np.diff(uni_ts)))]
    slopes = []
    index_start = 0
    for run, signature in ts_encode:
        index_end = index_start + run
        run_vals = uni_ts[index_start:index_end + 1]

        if signature == 1:  # it is an increase
            rise = run_vals.iloc[-1] - run_vals.iloc[0]  # last element - first element
            slope = rise / len(run_vals)
            slopes.append(slope)
        index_start = index_end
    if len(slopes) == 0:
        return np.float64(0)
    else:
        return np.float64(np.max(slopes))


def get_slope_of_longest_mono_decrease(uni_ts: Union[pd.Series, np.ndarray]) -> np.float64:
    """
    Identifies the longest monotonic decrease and gets the slope.

    :return: The slope of the longest monotonic decrease in `uni_ts`.
    """
    ts_encode = [(len(list(group)), name) for name, group in groupby(_sign(np.diff(uni_ts)))]
    slopes = []
    index_start = 0
    for run, signature in ts_encode:
        index_end = index_start + run
        run_vals = uni_ts[index_start:index_end + 1]

        if signature == -1:  # it is an increase
            rise = run_vals.iloc[-1] - run_vals.iloc[0]  # last element - first element
            slope = rise / len(run_vals)
            slopes.append(slope)
        index_start = index_end
    if len(slopes) == 0:
        return np.float64(0)
    else:
        return np.float64(np.max(slopes))


def get_avg_mono_increase_slope(uni_ts: Union[pd.Series, np.ndarray]) -> np.float64:
    """
    :return: The average slope of monotonically increasing segments.
    """
    ts_encode = [(len(list(group)), name) for name, group in groupby(_sign(np.diff(uni_ts)))]
    slopes = []
    index_start = 0
    for run, signature in ts_encode:
        index_end = index_start + run
        run_vals = uni_ts[index_start:index_end + 1]

        if signature == 1:  # it is an increase
            rise = run_vals.iloc[-1] - run_vals.iloc[0]  # last element - first element
            slope = rise / len(run_vals)
            slopes.append(slope)
        index_start = index_end
    if len(slopes) == 0:
        return np.float64(0)
    else:
        return np.float64(np.mean(slopes))


def get_avg_mono_decrease_slope(uni_ts: Union[pd.Series, np.ndarray]) -> np.float64:
    """
    :return: The average slope of monotonically decreasing segments.
    """
    ts_encode = [(len(list(group)), name) for name, group in groupby(_sign(np.diff(uni_ts)))]
    slopes = []
    index_start = 0
    for run, signature in ts_encode:
        index_end = index_start + run
        run_vals = uni_ts[index_start:index_end + 1]

        if signature == -1:  # it is an increase
            rise = run_vals.iloc[-1] - run_vals.iloc[0]  # last element - first element
            slope = rise / len(run_vals)
            slopes.append(slope)
        index_start = index_end
    if len(slopes) == 0:
        return np.float64(0)
    else:
        return np.float64(np.mean(slopes))


# -----------------------------------------------------------
#                       HELPER METHODS                      #
# -----------------------------------------------------------


def _difference_derivative(uni_ts: Union[pd.Series, np.ndarray], N: int = 1):
    if N < 1:
        return None
    else:
        return np.subtract(uni_ts[N:], uni_ts[:-N])


def _gradient_derivative(uni_ts: Union[pd.Series, np.ndarray]):
    return np.gradient(uni_ts)


def _sign(uni_ts: Union[pd.Series, np.ndarray]):
    return np.sign(uni_ts)
