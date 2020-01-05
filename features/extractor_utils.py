import pandas as pd
import features.feature_collection as fc


def get_methods_for_names(method_names: list):
    """
    for a given method-name, it finds it in `feature_collection` and returns it as a callable
    method.

    :param method_names: name of the method of interest that exists in `feature_collection`.
    :return: a callable instance of the method whose name is given.
    """
    callable_methods = []
    for m in method_names:
        try:
            callable_m = getattr(fc, m)
        except AttributeError as e:
            raise AttributeError(
                '''
                The statistical feature '{}' is invalid!
                Hint: To see all available features, run the following snippet:
                
                    import features.feature_collection as fc
                    help(fc)
                
                Any method-name starting with 'get_' can be used as a statistical feature.
                '''.format(m)
            )
        callable_methods.append(callable_m)
    return callable_methods


def calculate_one_mvts(df_mvts: pd.DataFrame, features_list: list) -> pd.DataFrame:
    """
    This method computes a list of F statistical features on the given multivariate time series
    of P parameters. The output is a dataframe of dimension P X F, that looks like::

        -----------------------
            f1    f2    ...
        p1  val   val   ...
        p2  val   val   ...
        ... ...   ...   ...
        -----------------------

    Note: The statistical features will be extracted from all the give columns. So, in case it is
    needed only over some of the time series, then only those selected columns should be passed in.

    :param df_mvts: a mvts dataframe from which the features are to be extracted.
    :param features_list : a list of all callable functions (from `features.feature_collection`)
           to be executed on the given mvts.
    :return: a dataframe with the parameters as rows, and statistical features as columns.
    """
    col_names = list(df_mvts)
    df_features = pd.DataFrame(index=col_names, dtype=float)
    for feature in features_list:
        feature_name = feature.__name__.replace('get_', '')
        df_extracted_feature = df_mvts.apply(feature, axis=0)
        df_features = df_features.assign(tmp=df_extracted_feature.values)
        df_features.rename(columns={'tmp': feature_name}, inplace=True)
    return df_features


def flatten_to_row_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    For a given dataframe of dimension P X F, where the row names (i.e., df's indices) are the
    time series (i.e., parameters') names, and the column names are the statistical features, this
    method flattens the given dataframe into a single-row dataframe of dimension 1 X (P X F). The
    columns names in the resultant dataframe is derived from the given dataframe df, by combining
    the row and column names of the given dataframe.

    For example, for a given df like the one below::

        -----------------------------------------------
            f1    f2    ...
        p1  val   val   ...
        p2  val   val   ...
        ... ...   ...   ...
        -----------------------------------------------

    the column names in the output dataframe would be::

        -----------------------------------------------
            P1_f1   P1_f2   ... P2_f1   P2_f2   ...
        1   val     val         val     val
        -----------------------------------------------

    :param df: the data frame to be flattened.
    :return: a dataframe with one row and P X F columns, with values similar to the given dataframe.
    """
    all_colnames = list(df)
    all_rownames = list(df.index)
    combined_names = [(str(y) + '_' + str(x)) for y in all_rownames for x in all_colnames]
    row = list(df.values.flatten())
    row_df = pd.DataFrame(columns=combined_names)
    row_df.loc[0] = row
    return row_df


def split(l: list, n_of_partitions: int) -> list:
    """
    Splits the given list l into n_of_paritions partitions of approximately equal size.

    :param l: the list to be split.
    :param n_of_partitions: number of partitions.
    :return: a list of the partitions, where each partition is a list itself.
    """
    k, m = divmod(len(l), n_of_partitions)
    return [l[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n_of_partitions)]