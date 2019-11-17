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

_summary_keywords: dict = {"params_col": 'Parameter-Name',
                           "null_col": "Null-Count",
                           "count_col": "Count",
                           "label_col": "Label"}

_5num_colnames: list = ['min', '25th', '50th', '75th', 'max']


class MVTSDataAnalysis:
    """
    This class walks through a directory of csv files (each being a mvts) and calculates
    estimated statistics of each of the parameters.

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

    def __init__(self, path_to_config):
        """
        This constructor initializes the class variables in order to use them in the methods for
        analysis of the MVTS dataset.

        :param path_to_config: path to the yml configuration file

        """

        path_to_config = os.path.join(CONST.ROOT, path_to_config)
        with open(path_to_config) as file:
            configs = yaml.load(file, Loader=yaml.FullLoader)

        self.path_to_dataset = os.path.join(CONST.ROOT, configs['PATH_TO_MVTS'])
        _, _, self.path_to_all_mvts = next(walk(self.path_to_dataset))
        self.mvts_parameters: list = configs['MVTS_PARAMETERS']
        self.summary = pd.DataFrame()

    def compute_summary(self, parameters_list: list = None, first_k: int = None):
        """
        By reading each csv file from the path listed in the configuration file, this method
        calculates all the basic statistics with respect to each parameter (each column of the
        mvts). As the data is distributed in several csv files this method computes the
        statistics on each mvts and updates the computed stats in a streaming fashion.

        **Note**: Computing the quantiles of parameters globally require loading the entire data
        on memory. To avoid this, we use `TDigest` data structure to estimate it, while loading
        one mvts at a time.

        As it calculates the statistics, it populates `self.summary` dataframe of the class with
        all the required data corresponding to each parameter. Below are the column names of the
        summary dataframe::
            - `Parameter Name`: Contains the timeseries' parameter name,
            - `Null Count`: Contains the number of null entries per parameter,
            - `Min`: Contains the minimum value of each parameter (without considering the null/nan
              values),
            - `Q1`: Contains the 1-st quartile (25%) of each parameter (without considering
              the null/nan values),
            - `Mean`: Contains the `mean` of each parameter (without considering the null/nan
              values),
            - `Median`: Contains the `median` of each parameter (without considering the
              null/nan values),
            - `Q3`: Contains the 3-rd quartile (75%) of each parameter (without considering
              the null/nan values),
            - `Max`: Contains the `min` value of each parameter (without considering the null/nan
              values)

        :param first_k: (Optional) If provided, only the fist `k` mvts will be processed. This is
                        mainly for getting some preliminary results in case the number of mvts
                        files is very large.
        :param parameters_list: (Optional) User may specify the list of parameters for which
                            statistical analysis is needed. If no parameters_list is provided by the
                            user then all existing numeric parameters are included in the list.
        :return: None.
        """

        if first_k is not None:
            all_csv_files = self.path_to_all_mvts[:first_k]
        else:
            all_csv_files = self.path_to_all_mvts
        n = len(all_csv_files)

        # If parameters_list is not provided all the physical parameters listed in the
        # configuration
        # file, with numeric datatype will be considered.
        if parameters_list is None:
            # read one csv as an example
            df = pd.read_csv(os.path.join(self.path_to_dataset, self.path_to_all_mvts[0]), sep='\t')
            # get the columns of interest
            df = pd.DataFrame(df[self.mvts_parameters], dtype=float)
            # get a list of numeric columns
            parameters_list = df.select_dtypes([np.number]).columns

        total_param = parameters_list.__len__()

        param_seq = [""] * total_param
        # TODO: Line below produces objects with similar id!! Is this OK? (object.__repr__(
        #  digests[0])
        digests = [TDigest() for i in range(total_param)]
        null_counts = [0] * total_param
        col_counts = [0] * total_param
        i = 0
        j = 0
        for f in all_csv_files:
            console_str = '-->\t[{}/{}] \t\t File: {}'.format(i, n, f)
            sys.stdout.write("\r" + console_str)
            sys.stdout.flush()
            # Only .csv files should be processed
            if f.lower().find('.csv') != -1:
                sys.stdout.flush()
                i += 1
                abs_path = os.path.join(self.path_to_dataset, f)
                df_mvts: pd.DataFrame = pd.read_csv(abs_path, sep='\t')

                # needs interpolation to make up for the nan values. Otherwise tDigest won't
                # be able to process the data.
                df_mvts = utils.interpolate_missing_vals(df_mvts)
                # keep the requested params only
                try:
                    df_req = pd.DataFrame(df_mvts[parameters_list]).select_dtypes([np.number])
                except:
                    raise ValueError(
                        """
                        Please check the parameter list.
                        """
                    )
                j = 0
                # Iterate the mvts by column, and compute tDigest on each column.
                for (param, series) in df_req.iteritems():
                    temp_null_count = int(series.isnull().sum())
                    temp_count = int(series.count())
                    null_counts[j] += temp_null_count
                    col_counts[j] += temp_count
                    if series.isnull().sum() != 0:
                        series = series.dropna()
                    if not series.empty:
                        series = np.array(series.values.flatten())
                        param_seq[j] = param
                        digests[j].batch_update(series)
                        digests[j].compress()

                    j += 1

        all_columns = _5num_colnames[:]
        all_columns.insert(0, _summary_keywords['null_col'])
        all_columns.insert(0, _summary_keywords['count_col'])
        all_columns.insert(0, _summary_keywords['params_col'])

        summary_stat_df = pd.DataFrame(columns=all_columns)

        for i in range(total_param):
            attname = param_seq[i]
            count_col = col_counts[i]
            col_miss = null_counts[i]
            col_min = col_Q1 = col_mean = col_Q3 = col_max = 0
            if digests[i]:
                col_min = digests[i].percentile(0)
                col_Q1 = digests[i].percentile(25)
                col_mean = digests[i].percentile(50)
                col_Q3 = digests[i].percentile(75)
                col_max = digests[i].percentile(100)

            summary_stat_df.loc[i] = [attname, count_col, col_miss, col_min, col_Q1, col_mean,
                                      col_Q3, col_max]

        if summary_stat_df.empty:
            raise ValueError(
                """
                Unable to get MVTS Data Analysis. Please check the parameter list or the dataset files.
                """
            )
        summary_stat_df.reset_index(inplace=True)
        summary_stat_df.drop(labels='index', inplace=True, axis=1)

        self.summary = summary_stat_df

    def get_number_of_mvts(self):
        """
        :return: the number of mvts files located at the root directory listed in the
        configuration file.
        """
        return len(self.path_to_all_mvts)

    def get_average_mvts_size(self):
        """
        :return: the average size (in bytes) of the mvts files located at the root directory
        listed in the configuration file.
        """
        all_sizes_in_bytes = []
        for f in self.path_to_all_mvts:
            if f.lower().find('.csv') != -1:
                f = os.path.join(f, self.path_to_dataset)
                all_sizes_in_bytes.append(os.stat(f).st_size)
        return np.mean(all_sizes_in_bytes)

    def get_total_mvts_size(self):
        """
        :return: the total size (in butes) of the mvts files located at the root directory
        listed in the configuration file.
        """
        all_sizes_in_bytes = []
        for f in self.path_to_all_mvts:
            if f.lower().find('.csv') != -1:
                f = os.path.join(f, self.path_to_dataset)
                all_sizes_in_bytes.append(os.stat(f).st_size)
        return np.sum(all_sizes_in_bytes)

    def print_stat_of_directory(self):
        """
        Prints a summary of the mvts files located at the root directory listed in the
        configuration file.
        :return: None.
        """
        print('----------------------------------------')
        print('Directory:\t\t\t\t\t{}'.format(self.path_to_dataset))
        print('Total number of mvts files:\t{}'.format(self.get_number_of_mvts()))
        print('Total size:\t\t\t\t\t{}'.format(size(self.get_total_mvts_size())))
        print('Total average:\t\t\t\t{}'.format(size(self.get_average_mvts_size())))
        print('----------------------------------------')

    def get_missing_values(self) -> pd.DataFrame:
        """
        Gets the missing values counts for each parameter in the mvts files.

        :return: a dataframe with two columns, namely the parameters names and the counts of the
        corresponding missing values.
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
        Gets the five-number summary of each parameter in the mvts files.

        :return: a dataframe where the rows are `min`, `25th`, `50th`, `75th`, `max` and the columns
        are the parameters of the given dataframe.
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
    path_to_config = os.path.join(CONST.ROOT, CONST.PATH_TO_CONFIG)

    mvts = MVTSDataAnalysis(path_to_config)
    mvts.print_stat_of_directory()

    mvts.compute_summary(first_k=50)
    mvts.summary_to_csv(output_path='.', file_name='mvts_data_analysis_3_params.csv')
    # todo where to keep this candidate_phys_parameters

    print(mvts.summary.columns)
    mvts.print_summary()
    print(mvts.get_five_num_summary())
    print(mvts.get_missing_values())


if __name__ == '__main__':
    main()
