import os
from os import path, makedirs
import pandas as pd
import numpy as np

_summary_keywords: dict = {"params_col": 'Feature-Name',
                           "null_col": "Null-Count",
                           "count_col": "Val-Count",
                           "label_col": "Label",
                           "population": "Population"}

_5num_colnames: list = ['mean', 'std', 'min', '25th', '50th', '75th', 'max']


class ExtractedFeaturesAnalysis:
    """
    This class is responsible for data analysis of the extracted statistical features. It takes
    the extracted features produced by `features.feature_extractor.py` (or
    `features.feature_extractor_parallel.py`) and provides some basic analytics as follows:
        * A histogram of classes,
        * The counts of the missing values,
        * A five-number summary for each extracted feature.

    These summaries can be stored in a CSV file as well.
    """

    def __init__(self, extracted_features_df: pd.DataFrame, exclude: list = None):
        """
        A constructor that initializes the class variables.

        :param extracted_features_df: the extracted features as it was produced by
                                      `FeatureExtractor` in `features.feature_extractor` or
                                      `FeatureExtractorParallel` in
                                      `features.feature_extractor_parallel.
        :param exclude: (Optional) a list of column-names indicating which columns should be
                        excluded from this analysis. All non-numeric columns will automatically
                        be removed. But this argument can be used to drop some numeric columns (
                        e.g., ID) whose numerical statistics makes no sense.
        """
        self.df = extracted_features_df
        self.summary = pd.DataFrame()
        self.excluded_colnames = exclude

    def compute_summary(self):
        """
        Using the extracted data this method calculates all the basic analysis with respect to
        each statistical feature (each column of `extracted_features_df`).

        It populates the summary dataframe of the class with all the required data corresponding
        to each feature.

        Below are the column names of the summary dataframe:
            * 'Feature Name': Contains the timeseries statistical feature name,
            * 'Non-null Count': Contains the number of non-null entries per feature,
            * 'Null Count': Contains the number of null entries per feature,
            * 'Min': Contains the minimum value of the feature(Without considering the null or
              nan value),
            * '25th': Contains the first quartile(25%) of the feature values(Without considering the
              null or nan  value),
            * 'Mean': Contains the mean of the feature values(Without considering the null/nan
              value),
            * '50th': Contains the median of the feature values(Without considering the null/nan
              value),
            * '75th': Contains the third quartile(75%) of the feature values(Without considering the
              null/nan value),
            * 'Max': Contains the minimum value of the feature(Without considering the null/nan
              value),
            * 'Std. Dev': Contains the standard deviation of the feature(Without considering the
              null/nan value)

        The computed summary will be stored in the class field `summary`.
        """
        df_desc = pd.DataFrame(self.df)

        # drop the columns that were requested to be excluded
        if self.excluded_colnames is not None:
            df_desc.drop(labels=self.excluded_colnames, inplace=True, axis=1)

        # drop any non-numeric column
        if not self.df.empty:
            df_desc = df_desc.describe(include=[np.number])
        else:
            raise ValueError(
                '''
                It seems that the given dataframe is empty. First run 
                `features.feature_extractor.py`.
                '''
            )

        if df_desc.empty:
            raise ValueError(
                '''
                It seems that in the given dataframe no numeric features are available. First run 
                `features.feature_extractor.py`.
                '''
            )

        df_desc = df_desc.T
        df_desc.insert(0, _summary_keywords['params_col'], df_desc.index)
        df_desc.insert(2, _summary_keywords['null_col'], self.df.isnull().sum())
        # New colnames: [Feature-Name, Val-Count, Null-Count, mean, std, min, 25th, 50th, 75th, max]
        df_desc.columns = [_summary_keywords['params_col'],
                           _summary_keywords['count_col'],
                           _summary_keywords['null_col']] + _5num_colnames
        df_desc.reset_index(inplace=True)
        df_desc.drop(labels='index', inplace=True, axis=1)
        self.summary = df_desc

    def get_class_population(self, label: str) -> pd.DataFrame:
        """
        Gets the per-class population of the original dataset.

        :param label: The column-name corresponding to the class_labels.
        :return: a dataframe of two columns; class_labels and class counts.
        """
        population_df = self.df[label].value_counts()
        population_df = population_df.to_frame(_summary_keywords['population'])
        population_df.insert(0, label, population_df.index)
        return population_df.reset_index(drop=True)

    def get_missing_values(self) -> pd.DataFrame:
        """
        Gets the missing-value counts for each extracted feature.

        :return: a dataframe of two columns; the extracted features (i.e., column names of
        `extracted_features_df`) and the missing-value counts.
        """
        if self.summary.empty:
            raise ValueError(
                """
                Execute `compute_summary` before getting the missing values.
                """
            )
        count_df = self.summary[[_summary_keywords['params_col'], _summary_keywords['null_col']]]
        return count_df.reset_index(drop=True)

    def get_five_num_summary(self) -> pd.DataFrame:
        """
        returns the seven number summary of each extracted feature. This method does not compute
        the statistics, but only returns what was already computed in `compute_summary` method.

        :return: a dataframe where the columns are [Feature-Name, mean, std, min, 25th, 50th, 75th,
        max] and each row corresponds to the statistics on one of the extracted features.
        """
        if self.summary.empty:
            raise ValueError(
                """
                Execute `compute_summary` before getting the five number summary.
                """
            )
        colname_copy = _5num_colnames.copy()  # copy: we don't want to change `_5num_colnames`
        colname_copy.insert(0, _summary_keywords['params_col'])
        five_num_df = self.summary[colname_copy]
        return five_num_df.reset_index(drop=True)

    def print_summary(self):
        """
        Prints the summary dataframe to the console.
        """
        if self.summary.empty:
            print(
                '''
                The summary is empty. The method `compute_summary` needs to be executed before 
                printing the results.
                '''
            )
        else:
            print(self.summary.to_string())

    def summary_to_csv(self, output_path, file_name):
        """
        Stores the summary statistics.

        :param output_path: path to where the summary should be stored.
        :param file_name: name of the csv file. If the extension is not given, `.csv` will be
                          appended to the given name.
        """
        if self.summary.empty:
            raise ValueError(
                '''
                Execute `compute_summary` before storing the results.
                '''
            )
        if not path.exists(output_path):
            makedirs(output_path)
        if not file_name.endswith('.csv'):
            file_name = '{}.csv'.format(file_name)

        out_file = os.path.join(output_path, file_name)
        self.summary.to_csv(out_file, sep='\t', header=True, index=False)
        print('Data Analysis of the extracted features is stored at [{}]'.format(out_file))


def main():
    import pandas as pd
    import CONSTANTS as CONST
    from normalizing import normalizer

    f_path = os.path.join(CONST.ROOT,
                          'data/extracted_features/extracted_features_parallel_3_pararams_4_features.csv')
    mvts_df = pd.read_csv(f_path, sep='\t')
    # Normalizer Test on extracted feature dataset
    # excluded_col = extracted_features_df.select_dtypes(exclude=np.number).columns.to_list()
    excluded_col = ['id']
    # df_norm = normalizer.negativeone_one_normalize(extracted_features_df,excluded_col)
    # df_norm = normalizer.robust_standardize(extracted_features_df,excluded_col)
    # df_norm = normalizer.standardize(extracted_features_df,excluded_col)
    df_norm = normalizer.zero_one_normalize(mvts_df, excluded_col)
    print(df_norm)

    # efa = ExtractedFeaturesAnalysis(mvts_df, excluded_col)
    # efa.compute_summary()
    # efa.print_summary()

    # d = efa.get_five_num_summary()
    # print(d[d['Feature-Name'] == 'TOTUSJH_median'].values)
    # print(list(d))
    # print(efa.get_class_population(label='lab'))
    # print(efa.get_five_num_summary())
    # print(efa.get_missing_values())
    # efa.summary_to_csv(CONST.ROOT, 'data/extracted_features/xxxxx.csv')


if __name__ == '__main__':
    main()
