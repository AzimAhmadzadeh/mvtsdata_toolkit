import sys
import os
from os import path, makedirs, walk
import pandas as pd
import numpy as np
from tdigest import TDigest
import multiprocessing as mp
import CONSTANTS as CONST
from features import extractor_utils

_summary_keywords: dict = {"params_col": 'Feature Name',
                           "null_col": "Null Count",
                           "count_col": "Count",
                           "label_col": "Label"}

_5num_colnames: list = ['min', '25%', '50%', '75%', 'max']


def get_missing_values(summary_df) -> pd.DataFrame:
    """
    Gets the missing value counts for each feature.

    :param summary_df: whole summary dataframe
    :return: a dataframe with two columns feature name and the counts of missing values.
    """
    if summary_df.empty:
        raise ValueError(
            """
            Execute `compute_summary` before getting the missing values.
            """
        )
    count_df = summary_df[[_summary_keywords["params_col"], _summary_keywords["null_col"]]]
    return count_df


def get_five_num_summary(summary_df) -> pd.DataFrame:
    """
    Gets the five number summary of each feature.

    :param summary_df: whole summary dataframe
    :return: a dataframe where the rows are [min, 25%, 50%, 75%, max] and the columns are the
             features in the given dataframe.
    """
    if summary_df.empty:
        raise ValueError(
            """
            Execute `compute_summary` before getting the five number summary.
            """
        )
    _5num_colnames.insert(0, _summary_keywords['params_col'])
    five_num_df = summary_df[_5num_colnames]
    return five_num_df


def print_summary(summary_df):
    """
    Prints the summary dataframe to the console.

    :param summary_df: whole summary dataframe
    """
    if summary_df.empty:
        print(
            '''
            The summary is empty. The method `compute_summary` needs to be executed before 
            printing the results.
            '''
        )
    else:
        print(summary_df.to_string())


def summary_to_csv(summary_df, output_path, file_name):
    """
    Stores the summary statistics.

    :param summary_df: whole summary dataframe
    :param output_path: path to where the summary should be stored.
    :param file_name: name of the csv file. If the extension is not given, '.csv' will be
           appended to the given name.
    :return:
    """
    if summary_df.empty:
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
    summary_df.to_csv(out_file, sep='\t', header=True, index=False)
    print('Data Analysis of the MVTS Dataset is stored in [{}]'.format(out_file))


class MVTSDataAnalysisParallel:
    """
    Note: The objective of this class is almost the same as 'MVTSDataAnalysis' from
    'mvts_data_analysis.py'.
    The only difference is that this is modified to be used with multiprocessing. To know how the
    parallelism is designed, see the main method in this module.

    This class walks through a directory of csv files (each being a mvts) and calculates
    estimated statistics of each of the features.

    It will perform the below tasks:
        1. Read each MVTS(.csv files) from the folder where the MVTS dataset is kept, i.e.,
           /pet_datasets/subset_partition3. Parameter path_to_root will be provided by the user
           in time of creating the instance of this class.
        2. Perform Exploratory Data Analysis(EDA) on the MVTS dataset:
            a. Histogram of classes
            b. Missing Value count
            c. Five-Number summary of each physical parameter(Estimated Values)
        3. Summary report can be saved in .CSV file in output folder,
           i.e., /pet_datasets/mvts_analysis using summary_to_csv() method.

    This class uses t-digest, a new data structure for accurate accumulation of rank-based
    statistics in distributed system. TDigest module is installed in order to use this data
    structure.
    """

    def __init__(self, path_of_mvts_files, feature_list: list = None):
        """
        This constructor initializes the class variables in order to use them in the methods for
        analysis of the MVTS dataset.

        :param path_to_dataset: folder location of the MVTS dataset
        :param feature_list:

        """

        self.path_of_mvts_files = path_of_mvts_files
        # If feature_list is not provided all the features with
        # numeric datatype will be considered
        if feature_list is None:
            abs_path = path_of_mvts_files[0]
            print(abs_path)
            df: pd.DataFrame = pd.read_csv(abs_path, sep='\t')
            feature_list = df.select_dtypes([np.number]).columns

        total_param = feature_list.__len__()

        self.param_seq = [""] * total_param
        self.digests = [TDigest() for i in range(total_param)]
        self.null_counts = [0] * total_param
        self.col_counts = [0] * total_param
        self.feature_list = feature_list
        self.total_param = total_param
        self.summary = pd.DataFrame()

    def compute_summary(self, proc_id: int, output_list: list, first_k: int = None):
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

        :param proc_id:
        :param output_list: a list (ListProxy) of the features (dataframes) extracted by each
               process.
        :param first_k: (Optional) If provided, then only the fist k mvts will be processed. This is
                        mainly for getting some preliminary results in case the number of mvts
                        files is very large.

        :return: dataframe with data analysis summary
        """

        if first_k is not None:
            all_csv_files = self.path_of_mvts_files[:first_k]
        else:
            all_csv_files = self.path_of_mvts_files
        n = len(all_csv_files)

        i = 0
        j = 0
        for f in all_csv_files:
            console_str = '\t[PID:{} --> {}/{}] \t\t File: {}'.format(proc_id, i, n, f)
            sys.stdout.write("\r" + console_str)
            sys.stdout.flush()
            # Only .csv file needs to be processed
            if f.lower().find('.csv') != -1:
                sys.stdout.flush()
                i += 1

                df_mvts: pd.DataFrame = pd.read_csv(f, sep='\t')
                # If there is any feature_list[] provided in time of instance creation of class MVTSDataAnalysisParallel
                # keep only the requested features from the whole dataset.

                try:
                    df_req = pd.DataFrame(df_mvts[self.feature_list]).select_dtypes([np.number])
                except:
                    # If the features provided in feature_list doesn't exists raise exception
                    raise ValueError(
                        """
                        Please check the feature_list[] provided to create instance of class MVTSDataAnalysisParallel .
                        The feature name doesn't match with the dataset column name.
                        """
                    )
                j = 0
                for (param, series) in df_req.iteritems():
                    temp_null_count = int(series.isnull().sum())
                    temp_count = int(series.count())
                    self.null_counts[j] += temp_null_count
                    self.col_counts[j] += temp_count
                    if series.isnull().sum() != 0:
                        series = series.dropna()
                    if not series.empty:
                        series = np.array(series.values.flatten())
                        self.param_seq[j] = param
                        self.digests[j].batch_update(series)
                        self.digests[j].compress()

                    j += 1
        all_columns = ['Feature Name', 'Count', 'Null Count', 'TDigest']
        eda_dict = pd.DataFrame(columns=all_columns)
        for i in range(0, j):
            attname = self.param_seq[i]
            count_col = self.col_counts[i]
            col_miss = self.null_counts[i]
            if self.digests[i]:
                col_digest = self.digests[i]
            eda_dict.loc[i] = [attname, count_col, col_miss, col_digest]

        output_list.append(eda_dict)


def main():
    """
    # TODO: This should NOT be in `main`, but in a method that users can actually use.
    
    Note: This module runs in parallel. Make sure that the settings are memory efficient
    before you run it.

    Settings:
        1. n_procs: is the number of processes to be employed. This number will be used to
        partition the data in a way that each process gets approximately the same number of
        files to extract features from.
        2. n_of_physical_params: is the index of the parameter (column in mvts csv files)
        indicating that all the parameters from index 1 to this index will be considered for
        feature extraction.For example, if it is set to 24, then first 24 columns will be
        considered only. (The timestamp which is the first column does not count.)

    """

    n_procs = 3  # total number of processes
    path_to_root = os.path.join(CONST.ROOT, 'pet_datasets/subset_partition3')
    output_filename = 'mvts_data_analysis_summary.csv'

    # ------------------------------------------------------------
    # Collect all files (only the absolute paths)
    # ------------------------------------------------------------
    all_files = list()
    dirpath, _, all_csv = next(walk(path_to_root))
    for f in all_csv:
        abs_path = path.join(dirpath, f)
        all_files.append(abs_path)

    # ------------------------------------------------------------
    # partition the files to be distributed among processes.
    # ------------------------------------------------------------
    partitions = extractor_utils.split(all_files, n_procs)
    # ------------------------------------------------------------
    # Assign a partition to each process
    # ------------------------------------------------------------
    proc_id = 0
    manager = mp.Manager()
    extracted_features = manager.list()
    jobs = []
    first_k = 10
    for partition in partitions:
        pc = MVTSDataAnalysisParallel(partition)
        process = mp.Process(target=pc.compute_summary,
                             args=(proc_id, extracted_features, first_k))
        jobs.append(process)
        proc_id = proc_id + 1

    for j in jobs:
        j.start()

    for j in jobs:
        j.join()

    print('A-----------> {}'.format(type(extracted_features)))
    print('B-----------> {}'.format(len(extracted_features)))
    print('C-----------> {}'.format(type(extracted_features[0])))

    df = extracted_features[0]
    null_count = df['Null Count']
    col_count = df['Count']
    t_digest = df['TDigest']
    for i in range(1, n_procs - 1):
        df_next = extracted_features[i]
        null_count += df_next['Null Count']
        col_count += df_next['Count']
        t_digest += df_next['TDigest']

    all_columns = _5num_colnames[:]
    five_sum = pd.DataFrame(columns=all_columns)
    i = 0

    for td in t_digest:
        five_sum.loc[i] = [td.percentile(0), td.percentile(25), td.percentile(50),
                           td.percentile(75), td.percentile(100)]
        i += 1

    df['Null Count'] = null_count
    df['Count'] = col_count
    df[all_columns] = five_sum
    df = df.drop(['TDigest'], axis=1)
    print_summary(df)
    path_to_output = os.path.join(CONST.ROOT, 'pet_datasets')
    summary_to_csv(df, path_to_output, output_filename)


if __name__ == '__main__':
    main()
