import sys
import os
from os import path, makedirs, walk
import pandas as pd
import numpy as np
from tdigest import TDigest
from hurry.filesize import size
import utils
import yaml
import CONSTANTS as CONST
from features import extractor_utils
from configs.config_reader import ConfigReader

_summary_keywords: dict = {"params_col": 'Parameter-Name',
                           "count_col": "Val-Count",
                           "null_col": "Null-Count",
                           "mean_col": "mean",
                           "tdigest_col": "TDigest"}

_5num_colnames: list = ['min', '25th', '50th', '75th', 'max']


def _evaluate_args(params_name: list, params_index: list):
    """
    This method throws an exception if both of `params_name` and `params_index` are provided.
    :param params_name:
    :param params_index:

    """
    has_param_name_in_arg = (params_name is not None) and (len(params_name) > 0)
    has_param_index_in_arg = (params_index is not None) and (len(params_index) > 0)

    if has_param_name_in_arg and has_param_index_in_arg:
        raise ValueError(
            """
            One and only one of the two arguments (params_name_list, params_index) must
            be provided.
            """
        )

    return True


def _unwrap_self_compute_summary(*arg, **kwarg):
    """
    This unwraps a class method so that it can perform independently and thus, can be called in
    parallel. More specifically, we need to be able to call `compute_summary` both sequentially and
    in parallel. In case of a parallel call, when it is called in a child process, the child process
    gets its own separate copy of the class instance. So, the class method needs to be unwrapped
    to a non-class method so that the class variables don't overlap.
    :param arg:
    :param kwarg:

    """
    return MVTSDataAnalysis.compute_summary(*arg, **kwarg)


class MVTSDataAnalysis:
    """
    This class walks through a directory of CSV files (each being an MVTS) and calculates
    estimated statistics of each of the parameters.

    It will perform the below tasks:
        1. Read each MVTS (.csv files) from the folder where the MVTS dataset is kept, i.e.,
           /pet_datasets/subset_partition3. Parameter path_to_root will be provided by the user
           at the time of creating the instance of this class.
        2. Perform Exploratory Data Analysis (EDA) on the MVTS dataset
            a. Histogram of classes
            b. Missing Value count
            c. Six-Number summary of each physical parameter(Estimated Values)
        3. Summary report can be saved in .CSV file in output folder,
           i.e., /pet_datasets/mvts_analysis using summary_to_csv() method.

    This class uses t-digest, a new data structure for accurate accumulation of rank-based
    statistics in the distributed system. TDigest module is installed in order to use this data
    structure.
    """

    def __init__(self, path_to_config):
        """
        This constructor initializes the class variables in order to use them in the methods for
        analysis of the MVTS dataset.

        :param path_to_config: Path to the yml configuration file
        """
        path_to_config = os.path.join(CONST.ROOT, path_to_config)

        cr = ConfigReader(path_to_config)
        configs = cr.read()

        self.path_to_dataset = os.path.join(CONST.ROOT, configs['PATH_TO_MVTS'])
        _, _, self.all_mvts_paths = next(walk(self.path_to_dataset))
        self.mvts_parameters: list = configs['MVTS_PARAMETERS']
        self.summary = pd.DataFrame()

    def compute_summary_in_parallel(self, n_jobs: int, params_name: list = None,
                                    params_index: list = None, first_k: int = None,
                                    verbose: bool = False):
        """
        This method calls `compute_summary` in parallel. For more details, see `compute_summary`'s
        documentation.

        :param n_jobs: The number of processes to be employed.
        :param params_name: (Optional) User may specify the list of parameters for which
                            statistical analysis is needed. If no params_name is provided by
                            the user then all existing numeric parameters are included in the
                            list.
        :param params_index: (Optional) User may specify the list of indices corresponding to the
                             parameters provided in the configuration file.
        :param first_k: (Optional) If provided, only the first `k` MVTS will be processed. This is
                        mainly for getting some preliminary results in case the number of MVTS
                        files is too large.
        :param verbose: Set it to True if you want to see more details as the function is running.
                        Default is False.
        """
        import multiprocessing as mp
        # ------------------------------------------------------------
        # Collect all files (only the absolute paths)
        # ------------------------------------------------------------
        dirpath, _, all_csv = next(walk(self.path_to_dataset))

        if first_k is not None:
            all_csv = all_csv[:first_k]

        all_files = [path.join(dirpath, f) for f in all_csv]

        # ------------------------------------------------------------
        # partition the files to be distributed among processes.
        # ------------------------------------------------------------
        partitions = extractor_utils.split(all_files, n_jobs)
        # ------------------------------------------------------------
        # Assign a partition to each process
        # ------------------------------------------------------------
        proc_id = 0
        manager = mp.Manager()
        summary_stats = manager.list()
        jobs = []
        for partition in partitions:
            process = mp.Process(target=_unwrap_self_compute_summary,
                                 kwargs=({'self': self,
                                          'params_name': params_name,
                                          'params_index': params_index,
                                          'first_k': first_k,
                                          'partition': partition,
                                          'proc_id': proc_id,
                                          'verbose': verbose,
                                          'output_list': summary_stats}))
            jobs.append(process)
            proc_id = proc_id + 1

        for j in jobs:
            j.start()

        for j in jobs:
            j.join()

        # -----------------------
        df: pd.DataFrame = summary_stats[0]

        null_count = df[_summary_keywords['null_col']]
        col_count = df[_summary_keywords['count_col']]
        col_means = df[_summary_keywords['mean_col']]
        t_digest = df[_summary_keywords['tdigest_col']]

        for i in range(1, n_jobs):
            df_next = summary_stats[i]
            null_count += df_next[_summary_keywords['null_col']]
            col_count += df_next[_summary_keywords['count_col']]
            col_means += df_next[_summary_keywords['mean_col']]
            t_digest += df_next[_summary_keywords['tdigest_col']]

        all_columns = _5num_colnames[:]
        five_sum = pd.DataFrame(columns=all_columns)
        i = 0

        for td in t_digest:
            five_sum.loc[i] = [td.percentile(0), td.percentile(25), td.percentile(50),
                               td.percentile(75), td.percentile(100)]
            i += 1

        df[_summary_keywords['null_col']] = null_count
        df[_summary_keywords['count_col']] = col_count
        df[_summary_keywords['mean_col']] = col_means / n_jobs
        df[all_columns] = five_sum
        df = df.drop([_summary_keywords['tdigest_col']], axis=1)
        self.summary = df

    def compute_summary(self, params_name: list = None, params_index: list = None,
                        first_k: int = None, partition: list = None, proc_id: int = None,
                        verbose: bool = False, output_list: list = None):
        """
        By reading each CSV file from the path listed in the configuration file, this method
        calculates all the basic statistics with respect to each parameter (each column of the
        MVTS). As the data is distributed in several CSV files, this method computes the
        statistics on each MVTS and updates the computed stats in a streaming fashion.

        **Note**: Computing the quantiles of parameters globally requires loading the entire data
        into memory. To avoid this, we use `TDigest` data structure to estimate it, while loading
        one MVTS at a time.

        As it calculates the statistics, it populates `self.summary` dataframe of the class with
        all the required data corresponding to each parameter. Below are the column names of the
        summary dataframe::
            - `Parameter Name`: Contains the time series parameter name,
            - `Val-Count`: Contains the count of the values of all processed time series,
            - `Null Count`: Contains the number of null entries per parameter,
            - `Min`: Contains the minimum value of each parameter (without considering the null/nan
              values),
            - `25th`: Contains the 1-st quartile (25%) of each parameter (without considering
              the null/nan values),
            - `Mean`: Contains the `mean` of each parameter (without considering the null/nan
              values),
            - `50th`: Contains the `median` of each parameter (without considering the
              null/nan values),
            - `75th`: Contains the 3-rd quartile (75%) of each parameter (without considering
              the null/nan values),
            - `Max`: Contains the `min` value of each parameter (without considering the null/nan
              values)

        :param first_k: (Optional) If provided, only the first `k` MVTS will be processed. This is
                        mainly for getting some preliminary results in case the number of MVTS
                        files is too large.
        :param params_name: (Optional) User may specify the list of parameters for which
                            statistical analysis is needed. If no params_name is provided by
                            the user then all existing numeric parameters are included in the
                            list.
        :param params_index: (Optional) User may specify the list of indices corresponding to the
                             parameters provided in the configuration file.
        :param partition: Only for internal use. Ignore this.
        :param proc_id: Only for internal use. Ignore this.
        :param verbose: Set it to True if you want to see more details as the function is running.
                        Default is False.
        :param output_list: Only for internal use. Ignore this.

        :return: None
        """
        is_parallel = False
        if proc_id is not None and output_list is not None:
            is_parallel = True

        # -----------------------------------------
        # Verify arguments
        # -----------------------------------------
        _evaluate_args(params_name, params_index)

        # -----------------------------------------
        # Get all files (or the first first_k ones).
        # -----------------------------------------
        all_csv_files = []
        if is_parallel:
            # Use the given `partition` instead of all csv files.
            all_csv_files = partition

        else:
            all_csv_files = self.all_mvts_paths
            if first_k is not None:
                # Note: If `fist_k` was used in parallel version, it should have already been taken
                # into account in `do_execution_in_parallel`. So, no need to do it again.
                all_csv_files = self.all_mvts_paths[:first_k]

        n = len(all_csv_files)

        # if parameters are provided through arguments, use them.
        if params_name is not None:
            self.mvts_parameters = params_name
        if params_index is not None:
            self.mvts_parameters = [self.mvts_parameters[i] for i in params_index]

        # -----------------------------------------
        # Drop any non-numeric columns from the columns of interest.
        # -----------------------------------------
        # read one csv as an example
        df = pd.read_csv(os.path.join(self.path_to_dataset, self.all_mvts_paths[0]), sep='\t')
        # get the columns of interest
        df = pd.DataFrame(df[self.mvts_parameters], dtype=float)
        # get a list of numeric column-names
        numeric_params_name = df.select_dtypes(include=[np.number]).columns

        total_params = len(numeric_params_name)

        param_seq = [""] * total_params

        digests = [TDigest() for i in range(total_params)]
        null_counts = np.array(np.zeros(total_params), dtype=int)
        col_counts = np.array(np.zeros(total_params), dtype=int)
        col_means = np.array(np.zeros(total_params), dtype=float)
        i = 1
        j = 0
        for f in all_csv_files:
            if verbose:
                if is_parallel:
                    print('\t[PID:{} --> {}/{}] \t\t File: {}'.format(proc_id, i, n, f))
                else:
                    console_str = '-->\t[{}/{}] \t\t File: {}'.format(i, n, f)
                    sys.stdout.write("\r" + console_str)
                    sys.stdout.flush()

            # Only .csv files should be processed
            if not f.lower().endswith('.csv'):
                continue

            i += 1
            abs_path = os.path.join(self.path_to_dataset, f)
            df_mvts: pd.DataFrame = pd.read_csv(abs_path, sep='\t')

            df_req = df_mvts[numeric_params_name]  # keep only the numeric columns!

            null_counts += df_req.isna().sum().to_list()
            col_counts += df_req.count().to_list()
            col_means += np.nanmean(df_req, axis=0).tolist()

            j = 0
            # Iterate over mvts by column, and compute tDigest on each column.
            for (param, series) in df_req.iteritems():
                if series.isnull().sum() != 0:
                    # Drop NA's since tDigest can't handle them.
                    series = series.dropna()
                if not series.empty:
                    series = np.array(series.values.flatten())
                    param_seq[j] = param
                    digests[j].batch_update(series)
                    digests[j].compress()
                j += 1
        # END OF LOOP OVER all_csv_files

        if not is_parallel:  # percentiles can be retrieved from tDigest objects
            all_columns = _5num_colnames[:]
            all_columns.insert(0, _summary_keywords['mean_col'])
            all_columns.insert(0, _summary_keywords['null_col'])
            all_columns.insert(0, _summary_keywords['count_col'])
            all_columns.insert(0, _summary_keywords['params_col'])

            summary_stat_df = pd.DataFrame(columns=all_columns)

            for i in range(total_params):
                attname = param_seq[i]
                col_count = col_counts[i]
                col_miss = null_counts[i]
                col_mean = col_means[i] / len(all_csv_files)
                col_min = col_q1 = col_q2 = col_q3 = col_max = 0
                if digests[i]:
                    col_min = digests[i].percentile(0)
                    col_q1 = digests[i].percentile(25)
                    col_q2 = digests[i].percentile(50)
                    col_q3 = digests[i].percentile(75)
                    col_max = digests[i].percentile(100)

                summary_stat_df.loc[i] = [attname, col_count, col_miss, col_mean,
                                          col_min, col_q1, col_q2, col_q3, col_max]
            # END OF FOR over mvts parameters

            if summary_stat_df.empty:
                raise ValueError(
                    """
                    Unable to get MVTS Data Analysis. Please check the parameter list or the dataset
                    files.
                    """
                )
            summary_stat_df.reset_index(inplace=True)
            summary_stat_df.drop(labels='index', inplace=True, axis=1)

            self.summary = summary_stat_df
            return

        else:  # parallel case: needs tDigest objects and not the percentiles.
            all_columns = _summary_keywords.values()
            summary_stat_df = pd.DataFrame(columns=all_columns)

            for i in range(total_params):
                attname = param_seq[i]
                col_count = col_counts[i]
                col_miss = null_counts[i]
                col_average = col_means[i] / len(all_csv_files)
                col_digest = None
                if digests[i]:
                    col_digest = digests[i]
                summary_stat_df.loc[i] = [attname, col_count, col_miss, col_average, col_digest]
            output_list.append(summary_stat_df)

    def get_number_of_mvts(self):
        """
        :return: The number of MVTS files located at the root directory listed in the
        configuration file.
        """
        return len(self.all_mvts_paths)

    def get_average_mvts_size(self):
        """
        :return: The average size (in bytes) of the MVTS files located at the root directory
        listed in the configuration file.
        """
        all_sizes_in_bytes = []
        for f in self.all_mvts_paths:
            if f.lower().endswith('.csv'):
                f_path = os.path.join(self.path_to_dataset, f)
                all_sizes_in_bytes.append(os.stat(f_path).st_size)
        return np.mean(all_sizes_in_bytes)

    def get_total_mvts_size(self):
        """
        :return: The total size (in bytes) of the MVTS files located at the root directory
        listed in the configuration file.
        """
        all_sizes_in_bytes = []
        for f in self.all_mvts_paths:
            if f.lower().find('.csv') != -1:
                f_path = os.path.join(self.path_to_dataset, f)
                all_sizes_in_bytes.append(os.stat(f_path).st_size)
        return np.sum(all_sizes_in_bytes)

    def print_stat_of_directory(self):
        """
        Prints a summary of the MVTS files located at the root directory listed in the
        configuration file.

        :return: None
        """
        print('----------------------------------------')
        print('Directory:\t\t\t{}'.format(self.path_to_dataset))
        print('Total no. of files:\t{}'.format(self.get_number_of_mvts()))
        print('Total size:\t\t\t{}'.format(size(self.get_total_mvts_size())))
        print('Total average:\t\t{}'.format(size(self.get_average_mvts_size())))
        print('----------------------------------------')

    def get_missing_values(self) -> pd.DataFrame:
        """
        Gets the missing values counts for each parameter in the MVTS files.

        :return: A dataframe with two columns, namely the parameter names and the counts of the
        corresponding missing values.
        """
        if self.summary.empty:
            raise ValueError(
                """
                The summary is empty. The method `compute_summary` needs to be executed before 
                getting the missing values.
                """
            )
        count_df = self.summary[[_summary_keywords["params_col"], _summary_keywords["null_col"]]]
        return count_df

    def get_six_num_summary(self) -> pd.DataFrame:
        """
        Gets the six-number summary of each parameter in the MVTS files.

        :return: A dataframe where the rows are `mean`, `min`, `25th`, `50th`, `75th`, `max` and
        the columns are the parameters of the given dataframe.
        """
        if self.summary.empty:
            raise ValueError(
                """
                The summary is empty. The method `compute_summary` needs to be executed before 
                getting the 6-num summary.
                """
            )
        colnames = _5num_colnames.copy()
        colnames.insert(0, _summary_keywords['params_col'])
        colnames.insert(1, _summary_keywords['mean_col'])
        six_num_df = self.summary[colnames]
        return six_num_df

    def print_summary(self):
        """
        Prints the summary dataframe to the console.
        """
        if self.summary.empty:
            raise ValueError(
                '''
                The summary is empty. The method `compute_summary` needs to be executed before 
                printing the results.
                '''
            )
        else:
            print()
            print(self.summary.to_string())

    def summary_to_csv(self, output_path, file_name):
        """
        Stores the summary statistics.

        :param output_path: Path to where the summary should be stored.
        :param file_name: Name of the CSV file. If the extension is not given, '.csv' will be
               appended to the given name.

        :return: None
        """
        if self.summary.empty:
            raise ValueError(
                '''
                The summary is empty. The method `compute_summary` needs to be executed before 
                saving it as a csv file.
                '''
            )
        if not path.exists(output_path):
            makedirs(output_path)
        if not file_name.endswith('.csv'):
            file_name = '{}.csv'.format(file_name)

        out_file = os.path.join(output_path, file_name)
        self.summary.to_csv(out_file, sep='\t', header=True, index=False)
        print('Data Analysis of the MVTS Dataset is stored in [{}]'.format(out_file))
