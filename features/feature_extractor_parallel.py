import os
import pandas as pd
from os import path, walk
import utils
import yaml
import CONSTANTS as CONST
from features import extractor_utils


class FeatureExtractorParallel:
    """
    #TODO: This documentation needs a carefull review.

    Note: The objective of this class is exactly the same as 'FeatureExtractor' from
    'feature_extractor.py'.
    The only difference is that this is modified to be used with multiprocessing. To know how the
    parallelism is designed, see the main method in this module.

    This class walks through a directory of csv files of multivariate time series (mvts),
    and for each of them, computes all listed statistical parameters on each of the time series.
    In the mvts files, each column is one time series.

    Key points about the input and output data:
        - Input multivariate time series: 40 X 55 (length of ts, number of ts)
        - Important multivariate time series: 40 X 33 (length of ts, number of important ts)
        - Currently used multivariate time series: 40 X 24
        - Number of currently used physical parameters: 24
        - Number of statistical features: F
        - Number of csv files (slices of time series): N
        - Matrix of all features (df_all_features) extracted from one multivariate timeseries: N X (
          24*F)

    Example::
        import multiprocessing

        proc_id = 0
        manager = multiprocessing.Manager()
        extracted_features = manager.list()
        jobs = []
        for partition in partitions:  # one partition per process
            pc = FeatureExtractorParallel(features_list=features_list,
                                          params_name_list=params_name_list,
                                          params_index=None,
                                          need_interp=True)
            process = multiprocessing.Process(target=pc.calculate_all,
                                              args=(proc_id, partition, extracted_features))
            jobs.append(process)
            proc_id = proc_id + 1

        for j in jobs:
            j.start()

        for j in jobs:
            j.join()
    """

    def __init__(self, path_to_config: str):
        """
        This constructor simply initializes the following information to be prepared for
        extracting the statistical features from the physical parameters.

        :param path_to_config: path to where the corresponding configuration (YAML) file is located.
        """

        with open(path_to_config) as file:
            configs = yaml.load(file, Loader=yaml.FullLoader)

        # self.path_to_root = os.path.join(CONST.ROOT, configs['PATH_TO_MVTS']) " Not needed
        self.path_to_output = os.path.join(CONST.ROOT, configs['PATH_TO_EXTRACTED_FEATURES'])
        self.statistical_features: list = \
            extractor_utils.get_methods_for_names(configs['STATISTICAL_FEATURES'])
        self.mvts_parameters: list = configs['MVTS_PARAMETERS']
        self.metadata_tags: list = configs['META_DATA_TAGS']
        self.df_all_features = pd.DataFrame()

    def calculate_all(self, proc_id: int, all_csvs: list, output_list: list,
                      params_index_list: list = None,
                      need_interp: bool = True):
        """
        #Todo check params_index whether needed or not
        Computes all the give statistical features on each of the csv files in 'path_to_root'
        and stores the result in the form of a single csv file.

        If n processes are utilized, the list will contain n dataframes.

        :param proc_id: id of the process running this method.
        :param all_csvs: a list of the absolute paths to all csv files from which the features are
               to be extracted.
        :param output_list: a list (ListProxy) of the features (dataframes) extracted by each
               process.
        :param need_interp:
        :param params_index_list:

        :return: a csv file where each row corresponds to a time series and represents all the
        extracted features from that time series.
        """

        # -----------------------------------------
        # Verify arguments
        # -----------------------------------------
        has_param_name_arg = (self.mvts_parameters is not None) and (len(self.mvts_parameters) > 0)
        has_param_index_arg = (params_index_list is not None) and (len(params_index_list) > 0)
        if has_param_name_arg == has_param_index_arg:  # mutual exclusive
            raise ValueError(
                """
                One and only one of the two arguments (params_name_list, params_index) must
                be provided.
                """
            )

        if len(self.statistical_features) == 0:
            raise ValueError(
                """
                The argument 'self.statistical_features' cannot be empty!
                """
            )

        n = len(all_csvs)
        n_features = len(self.statistical_features)
        i = 1

        print('\n\n\t--------------PID--{}-------------------'.format(proc_id))
        print('\t\tTotal No. of Features:\t\t{}'.format(n_features))
        print('\t\tTotal No. of time series:\t{}'.format(n))
        print('\t\tOutput TS dimensionality ({} X {}):\t\t{}'.format(n, n_features, n * n_features))
        print('\t-----------------------------------------\n'.format())

        for f in all_csvs:
            print('\t PID:{} --> Total Processed: {} / {}'.format(proc_id, i, n))

            df_mvts: pd.DataFrame = pd.read_csv(f, sep='\t')

            # -----------------------------------------
            # Keep the important columns only
            # -----------------------------------------
            df_raw = pd.DataFrame()
            if has_param_name_arg:
                df_raw = pd.DataFrame(df_mvts[self.mvts_parameters], dtype=float)
            elif has_param_index_arg:
                df_raw = pd.DataFrame(df_mvts.iloc[:, self.params_index_list], dtype=float)

            # -----------------------------------------
            # Interpolate to get rid of the NaN values.
            # -----------------------------------------
            if need_interp:
                df_raw = utils.interpolate_missing_vals(df_raw)

            # -----------------------------------------
            # Extract all the features from each column of mvts.
            # -----------------------------------------
            extractedfeatures_df = extractor_utils.calculate_one_mvts(df_raw, self.statistical_features)

            # -----------------------------------------
            # Extract the given meta data from this mvts name.
            # -----------------------------------------
            tags_dict = dict()
            for tag in self.metadata_tags:
                tags_dict.update({tag: utils.extract_tagged_info(f, tag)})
            # -----------------------------------------
            # Flatten the resultant dataframe and add the mvts_id, class label, and start-time
            # and end-time.
            # row_df will then have these columns:
            #   ID | LAB | ST | ET | FEATURE_1 | ... | FEATURE_n
            # -----------------------------------------
            row_dfs = []
            for tag, extracted_info in tags_dict.items():
                row_dfs.append(pd.DataFrame({tag: [extracted_info]}))

            features_df = extractor_utils.flatten_to_row_df(extractedfeatures_df)
            row_dfs.append(features_df)
            row_df = pd.concat(row_dfs, axis=1)
            # -----------------------------------------
            # Append this row to 'df_all_features'
            # -----------------------------------------
            # if this is the first file, create the main dataframe, i.e., 'df_all_features'
            if i == 1:
                self.df_all_features = pd.DataFrame(row_df)
            else:
                # add this row to the end of the dataframe 'df_all_features'
                self.df_all_features = self.df_all_features.append(row_df)
            i = i + 1

        print('\n\t^^^^^^^^^^^^^^^^^^^^PID: {0}^^^^^^^^^^^^^^^^^^^^^'.format(proc_id))
        print('\n\tDone! {} files have been processed.'.format(i - 1))
        print('\tIn total, a dataframe of dimension {} X {} is created.'.format(
            self.df_all_features.shape[0],
            self.df_all_features.shape[1]))

        output_list.append(self.df_all_features)


def main():
    """
    TODO: this should not be in the `main` method but in a method that users can actually call.

    Note: This module runs in parallel. Make sure that the settings are memory efficient
    before you run it.

    Settings:
        1. n_procs: is the number of processes to be employed. This number will be used to
        partition the data in a way that each process gets approximately the same number of
        files to extract features from.
        2. n_of_physical_params: is the index of the last parameter (column in mvts csv files)
        indicating that all the parameters from index 1 to this index will be considered for
        feature extraction.For example, if it is set to 24, then first 24 columns will be
        considered only. (The timestamp which is the first column does not count.)
        3. features_list: is a collection of all the feature-extractors. The number of columns
        in the final results will depend on this list, since:
            |attributes of final results| = len(features_list) X n_of_physical_params

    :return:
    """

    import multiprocessing as mp

    n_procs = 2  # total number of processes
    # Todo: Check (as path_to_root is before instance creation)
    path_to_root = os.path.join(CONST.ROOT, 'pet_datasets/subset_partition3')
    output_filename = 'raw_features_p3_FL_parallel_with_conversion.csv'

    # ------------------------------------------------------------
    # Collect all files (only the absolute paths)
    # ------------------------------------------------------------
    all_files = list()
    dirpath, _, all_csv = next(walk(path_to_root))
    path_to_config = os.path.join(CONST.ROOT, CONST.PATH_TO_CONFIG)
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
    for partition in partitions:
        pc = FeatureExtractorParallel(path_to_config)
        process = mp.Process(target=pc.calculate_all,
                             args=(proc_id, partition, extracted_features))
        jobs.append(process)
        proc_id = proc_id + 1

    for j in jobs:
        j.start()

    for j in jobs:
        j.join()

    print('A-----------> {}'.format(type(extracted_features)))
    print('B-----------> {}'.format(len(extracted_features)))
    print('C-----------> {}'.format(type(extracted_features[0])))
    #
    print('All {} processes have finished their tasks.'.format([n_procs]))
    # -----------------------------------------
    # Store the csv of all features to 'path_to_dest'.
    # -----------------------------------------
    # Todo: Check (as instance still retains path_to_output should this be used?)
    if not os.path.exists(pc.path_to_output):
        os.makedirs(pc.path_to_output)

    fname = os.path.join(pc.path_to_output, output_filename)

    all_extracted_features = pd.concat(extracted_features)

    all_extracted_features.to_csv(fname, sep='\t', header=True, index=False)
    print('\n\tThe dataframe is stored at: {}'.format(fname))


if __name__ == '__main__':
    main()
