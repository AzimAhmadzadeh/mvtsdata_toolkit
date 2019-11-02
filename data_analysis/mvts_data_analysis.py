import sys
import os
from os import path, makedirs, walk
import pandas as pd
import numpy as np
from tdigest import TDigest
from hurry.filesize import size

import CONSTANTS as CONST

_summary_keywords: dict = {"params_col": 'Feature-Name',
                           "null_col": "Null-Count",
                           "count_col": "Count",
                           "label_col": "Label"}

_5num_colnames: list = ['min', '25%', '50%', '75%', 'max']


class MVTSDataAnalysis:
    """
    This class walks through a directory of csv files (each being a mvts) and calculates
    estimated statistics of each of the features.
    It will perform the below tasks:
        1. Read each MVTS(.csv files) from the folder where the MVTS dataset is kept, i.e.,
           /pet_datasets/subset_partition3. Parameter path_to_root will be provided by the user
           in time of creating the instance of this class.
        2. Perform Exploratory Data Analysis(EDA) on the MVTS dataset
            a. Histogram of classes
            b. Missing Value count
            c. Five-Number summary of each physical parameter(Estimated Values)
        3. Summary report can be saved in .CSV file in output folder,
        i.e., /pet_datasets/mvts_analysis using summary_to_csv() method.

    This class uses t-digest, a new data structure for accurate accumulation of rank-based
    statistics in distributed system. TDigest module is installed in order to use this data
    structure.
    """

    def __init__(self, path_to_dataset):
        """
        This constructor initializes the class variables in order to use them in the methods for
        analysis of the MVTS dataset.

        :param path_to_dataset: folder location of the MVTS dataset

        """

        path_to_dataset = os.path.join(CONST.ROOT, path_to_dataset)
        self.path_to_dataset, _, self.path_to_all_mvts = next(walk(path_to_dataset))
        self.summary = pd.DataFrame()

    def get_number_of_mvts(self):
        return len(self.path_to_all_mvts)

    def get_average_mvts_size(self):
        all_sizes_in_bytes = []
        for f in self.path_to_all_mvts:
            if f.lower().find('.csv') != -1:
                f = os.path.join(f, self.path_to_dataset)
                all_sizes_in_bytes.append(os.stat(f).st_size)
        return np.mean(all_sizes_in_bytes)

    def get_total_mvts_size(self):
        all_sizes_in_bytes = []
        for f in self.path_to_all_mvts:
            if f.lower().find('.csv') != -1:
                f = os.path.join(f, self.path_to_dataset)
                all_sizes_in_bytes.append(os.stat(f).st_size)
        return np.sum(all_sizes_in_bytes)

    def print_stat_of_directory(self):
        print('----------------------------------------')
        print('Directory:\t\t\t\t\t{}'.format(self.path_to_dataset))
        print('Total number of mvts files:\t{}'.format(self.get_number_of_mvts()))
        print('Total size:\t\t\t\t\t{}'.format(size(self.get_total_mvts_size())))
        print('Total average:\t\t\t\t{}'.format(size(self.get_average_mvts_size())))
        print('----------------------------------------')

    def compute_summary(self, first_k: int = None, feature_list: list = None):
        """
        By reading each CSV file from the MVTS dataset this method calculates all the basic analysis
        with respect to each feature(each column of csv). As the data is distributed in several
        csv files this method reads each csv and accumulate the values.

        t-digest data structure is used in order to get the quartiles.

        It populates the summary dataframe of the class with all the required data corresponding
        to each feature. This statistics are based on the overall MVTS dataset.

        Below are the column names of the summary dataframe,
            - 'Feature Name': Contains the timeseries feature name,
            - 'Null Count': Contains the number of null entries per feature,
            - 'Min': Contains the minimum value of the feature(Without considering the null/nan
              value),
            - 'Q1': Contains the first quartile(25%) of the feature values(Without considering
              the null/nan  value),
            - 'Mean': Contains the mean of the feature values(Without considering the null/nan
              value),
            - 'Median': Contains the median of the feature values(Without considering the
              null/nan  value),
            - 'Q3': Contains the third quartile(75%) of the feature values(Without considering
              the null/nan  value),
            - 'Max': Contains the minimum value of the feature(Without considering the null/nan
              value)

        :param first_k: (Optional) If provided, then only the fist k mvts will be processed. This is
                        mainly for getting some preliminary results in case the number of mvts
                        files is very large.
        :param feature_list: (Optional) User may specify the list of features for which statistical
                            analysis is needed. If no feature_list is provided by the user
                            then all existing numeric features are included in the list.
        :return: dataframe with data analysis summary
        """

        if first_k is not None:
            all_csv_files = self.path_to_all_mvts[:first_k]
        else:
            all_csv_files = self.path_to_all_mvts
        n = len(all_csv_files)

        # If feature_list is not provided all the physical parameter with
        # numeric datatype will be considered
        if feature_list is None:
            abs_path = os.path.join(self.path_to_dataset, all_csv_files[0])
            df: pd.DataFrame = pd.read_csv(abs_path, sep='\t')
            feature_list = df.select_dtypes([np.number]).columns

        total_param = feature_list.__len__()

        param_seq = [str for i in range(total_param)]
        digest = [TDigest() for i in range(total_param)]
        null_count = [0] * total_param
        col_count = [0] * total_param
        i = 0
        j = 0
        for f in all_csv_files:
            console_str = '-->\t[{}/{}] \t\t File: {}'.format(i, n, f)
            sys.stdout.write("\r" + console_str)
            sys.stdout.flush()
            # Only .csv file needs to be processed
            if f.lower().find('.csv') != -1:
                sys.stdout.flush()
                i += 1

                abs_path = os.path.join(self.path_to_dataset, f)
                df_mvts: pd.DataFrame = pd.read_csv(abs_path, sep='\t')
                # keep the requested params only
                try:
                    df_req = pd.DataFrame(df_mvts[feature_list]).select_dtypes([np.number])
                except:
                    raise ValueError(
                        """
                        Please check the parameter list.
                        """
                    )
                j = 0
                for (param, series) in df_req.iteritems():
                    temp_null_count = int(series.isnull().sum())
                    temp_count = int(series.count())
                    null_count[j] += temp_null_count
                    col_count[j] += temp_count
                    if series.isnull().sum() != 0:
                        series = series.dropna()
                    if not series.empty:
                        series = np.array(series.values.flatten())
                        param_seq[j] = param
                        digest[j].batch_update(series)
                        digest[j].compress()

                    j += 1

        all_columns = _5num_colnames[:]
        all_columns.insert(0, _summary_keywords['null_col'])
        all_columns.insert(0, _summary_keywords['count_col'])
        all_columns.insert(0, _summary_keywords['params_col'])

        eda_dict = pd.DataFrame(columns=all_columns)

        for i in range(0, j):
            attname = param_seq[i]
            count_col = col_count[i]
            col_miss = null_count[i]
            col_min = col_Q1 = col_mean = col_Q3 = col_max = 0
            if digest[i]:
                col_min = digest[i].percentile(0)
                col_Q1 = digest[i].percentile(25)
                col_mean = digest[i].percentile(50)
                col_Q3 = digest[i].percentile(75)
                col_max = digest[i].percentile(100)

            eda_dict.loc[i] = [attname, count_col, col_miss, col_min, col_Q1, col_mean, col_Q3,
                               col_max]

        if eda_dict.empty:
            raise ValueError(
                """
                Unable to get MVTS Data Analysis. Please check the parameter list or the dataset files.
                """
            )
        eda_dict.reset_index(inplace=True)
        eda_dict.drop(labels='index', inplace=True, axis=1)

        self.summary = eda_dict

    def get_missing_values(self) -> pd.DataFrame:
        """
        Gets the missing value counts for each feature.
        :return: a dataframe with two columns feature name and the counts of missing values.
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
        Gets the five number summary of each feature.

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
        :return:
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
        print('Data Analysis of the MVTS Dataset is stored in [{}]'.format(out_file))


def main():
    path_to_dataset = os.path.join(CONST.ROOT, 'pet_datasets/subset_partition3')
    mvts = MVTSDataAnalysis(path_to_dataset)
    mvts.print_stat_of_directory()

    mvts.compute_summary(first_k=50)
    # mvts.compute_summary(CONST.CANDIDATE_PHYS_PARAMETERS)
    # todo where to keep this candidate_phys_parameters

    print(mvts.summary.columns)
    mvts.print_summary()
    print(mvts.get_five_num_summary())
    print(mvts.get_missing_values())


if __name__ == '__main__':
    main()
