import pandas as pd


def interpolate_missing_vals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolates the missing values (i.e., NaN's) using linear interpolation. That
    is, for any group of consecutive missing values, it treats the values as equally
    spaced numbers between the present values before and after the gap.
    This does not impact non-numerical values.

    :return: The interpolated version of the given dataframe.
    """
    if df.isna().sum().sum() != 0:
        return df.interpolate(method='linear', axis=0, limit_direction='both')
    else:
        return df
