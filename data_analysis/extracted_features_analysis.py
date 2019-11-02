import os
from os import path, makedirs
import pandas as pd
import numpy as np

_summary_keywords: dict = {"params_col": 'Feature Name',
                           "null_col": "Null Count",
                           "count_col": "Count",
                           "label_col": "Label"}

_5num_colnames: list = ['min', '25%', '50%', '75%', 'max']


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

    def __init__(self, mvts_df: pd.DataFrame, exclude: list):
        """
        A constructor that initializes the class variables.

        :param mvts_df: the extracted features as it was produced by `features.feature_extractor.py`
                        (or `features.feature_extractor_parallel.py`).
        """
        self.df = mvts_df
        self.summary = pd.DataFrame()
        self.excluded_colnames = exclude

    def compute_summary(self):
        """
        Using the extracted data (self.mvts_df-->extracted_feature.csv) this method calculates
        all the basic analysis with respect to each statistical feature(each column of
        mvts_df).

        It populates the summary dataframe of the class with all the required data corresponding
        to each feature.

        Below are the column names of the summary dataframe:
            * 'Feature Name': Contains the timeseries statistical feature name,
            * 'Non-null Count': Contains the number of non-null entries per feature,
            * 'Null Count': Contains the number of null entries per feature,
            * 'Min': Contains the minimum value of the feature(Without considering the null or
              nan value),
            * 'Q1': Contains the first quartile(25%) of the feature values(Without considering the
              null or nan  value),
            * 'Mean': Contains the mean of the feature values(Without considering the null/nan
              value),
            * 'Median': Contains the median of the feature values(Without considering the null/nan
              value),
            * 'Q3': Contains the third quartile(75%) of the feature values(Without considering the
              null/nan value),
            * 'Max': Contains the minimum value of the feature(Without considering the null/nan
              value),
            * 'Std. Dev': Contains the standard deviation of the feature(Without considering the
              null/nan value)

        :return: dataframe with data analysis summary
        """

        if not self.df.empty:
            df_desc = self.df.describe(include=[np.number])
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

        df_desc.drop(labels=self.excluded_colnames, inplace=True, axis=1)
        df_desc = df_desc.T
        df_desc.insert(0, _summary_keywords['params_col'], df_desc.index)
        df_desc.insert(2, _summary_keywords['null_col'], self.df.isnull().sum())
        df_desc.reset_index(inplace=True)
        df_desc.drop(labels='index', inplace=True, axis=1)
        self.summary = df_desc

    def get_class_population(self, label: str) -> pd.DataFrame:
        """
        Gets the per-class population of the original dataset.

        :param label: Column name corresponding to the labels.
        :return: a dictionary of labels (as keys) and class populations (as values).
        """
        population_df = self.df[label].value_counts()
        population_df = population_df.to_frame('Count')
        population_df.insert(0, 'Label', population_df.index)
        return population_df.set_index('Label')

    def get_missing_values(self) -> pd.DataFrame:
        """
        Gets the missing value counts for each feature.

        :return: a dictionary of column names (as keys) and the counts of missing values (as
        values).
        """
        if self.summary.empty:
            raise ValueError(
                """
                Execute `compute_summary` before getting the missing values.
                """
            )
        count_df = self.summary[[_summary_keywords["params_col"], _summary_keywords["null_col"]]]
        return count_df

    def get_five_num_summary(self) -> pd.DataFrame:
        """
        Gets the five number summary of each feature. This method does not compute the five-number
        statistics. It only returns what was already computed in `compute_summary` method.

        :return: a dataframe where the rows are [min, 25%, 50%, 75%, max] and the columns are the
                 features in the given dataframe.
        """
        if self.summary.empty:
            raise ValueError(
                """
                Execute `compute_summary` before getting the five number summary.
                """
            )
        _5num_colnames.insert(0, _summary_keywords['params_col'])
        five_num_df = self.summary[_5num_colnames]
        return five_num_df

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
        :param file_name: name of the csv file. If the extension is not given, '.csv' will be
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
    from utils import normalizer

    f_path = os.path.join(CONST.ROOT,
                          'pet_datasets/extracted_features/non_unittest_extracted_features.csv')
    mvts_df = pd.read_csv(f_path, sep='\t')
    # Normalizer Test on extracted feature dataset
    # excluded_col = mvts_df.select_dtypes(exclude=np.number).columns.to_list()
    excluded_col = ['id']  # .insert(0,'id')
    # df_norm = normalizer.negativeone_one_normalize(mvts_df,excluded_col)
    # df_norm = normalizer.robust_standardize(mvts_df,excluded_col)
    # df_norm = normalizer.standardize(mvts_df,excluded_col)
    df_norm = normalizer.zero_one_normalize(mvts_df, excluded_col)
    print(df_norm)

    efa = ExtractedFeaturesAnalysis(mvts_df, ['id'])
    efa.compute_summary()
    efa.print_summary()
    print(efa.get_class_population(label='lab'))
    print(efa.get_five_num_summary())
    print(efa.get_missing_values())
    efa.summary_to_csv(CONST.ROOT, 'summary.csv')


if __name__ == '__main__':
    main()
