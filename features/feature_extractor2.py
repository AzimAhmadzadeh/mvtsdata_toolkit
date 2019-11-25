import pandas as pd
import os
import sys
from os import path, walk
import utils
from features import extractor_utils
import yaml
import CONSTANTS as CONST
import time


def _evaluate_args(params_name: list, params_index: list,
                   features_list: list, feature_index: list):
    """
    This method throws an exception if both of `params_name` and `params_index`, or both of
    `features_list` and features_index` are provided.
    :param params_name:
    :param params_index:
    :param features_list:
    :param feature_index:
    :return:
    """
    has_param_name_in_arg = (params_name is not None) and (len(params_name) > 0)
    has_param_index_in_arg = (params_index is not None) and (len(params_index) > 0)
    has_feature_name_in_arg = (features_list is not None) and (len(features_list) > 0)
    has_feature_index_in_arg = (feature_index is not None) and (len(feature_index) > 0)

    if has_param_name_in_arg and has_param_index_in_arg:
        raise ValueError(
            """
            One and only one of the two arguments (params_name_list, params_index) must
            be provided.
            """
        )
    if has_feature_name_in_arg and has_feature_index_in_arg:
        raise ValueError(
            """
            Either in the configuration file or by the argument `feature_list`, a list of 
            statistical features must be provided!
            """
        )
    return True


def unwrap_self_do_extraction(*arg, **kwarg):
    """
    This unwraps a class method so that it can perform independently and thus, can be called in
    parallel.
    :param arg:
    :param kwarg:
    :return:
    """
    return FeatureExtractor.do_extraction(*arg, **kwarg)


class FeatureExtractor:
    """
    This class loads the configuration file `feature_extraction_configs.yml`, and reads the
    following pieces of information from it.

    Below are the column names of the summary dataframe:
        * `PATH_TO_MVTS`: path to where the csv (mvts) files are stored.
        * `MVTS_PARAMETERS`: a list of time series name; only those listed here will be processed.
        * `STATISTICAL_FEATURES`: a list of statistical features to be computed on each time
           series.
        * `META_DATA_TAGS`: a list of tags used in the mvts file names; to be used for extraction of
           some metadata from file names.
        * `PATH_TO_EXTRACTED_FEATURES`: path to a directory where the extracted features (one csv
           file) will be stored.

    Based on these values, it walks through the directory `PATH_TO_MVTS` and for each of the mvts
    files, it computes the statistical features listed in `STATISTICAL_FEATURES` on all time series
    listed in `MVTS_PARAMETERS`. It uses the tags in `META_DATA_TAGS` to extract some metadata,
    such as class `label`, `time stamp`, `id`, etc.

    The resultant dataframe (i.e., the extracted features) will have `T X F + x` columns, where `F`
    is the total number of features (i.e., `len(STATISTICAL_FEATURES)`), `T` is the total number of
    time series parameters (i.e., `len(MVTS_PARAMETERS)`), and `x` is the number of meta data
    extracted from the file names (i.e., `len(META_DATA_TAGS)`).

    In the extracted features dataframe, the column-name of the nominal attributes is of the
    following structure::

        <TIME_SERIES_NAME>_<statistic_name>

    For instance, for a time series named `DENSITY` and the statistical feature `mean`, the
    corresponding column-name would be `DENSITY_mean`.
    """

    def __init__(self, path_to_config: str):
        """
        This constructor loads all necessary information from the config file provided by
        `path_to_config`.

        :param path_to_config: path to where the corresponding configuration (YAML) file is located.
        """
        with open(path_to_config) as file:
            configs = yaml.load(file, Loader=yaml.FullLoader)
        self.path_to_config = path_to_config
        self.path_to_root = os.path.join(CONST.ROOT, configs['PATH_TO_MVTS'])
        self.path_to_output = os.path.join(CONST.ROOT, configs['PATH_TO_EXTRACTED_FEATURES'])
        self.statistical_features: list = configs['STATISTICAL_FEATURES']
        self.mvts_parameters: list = configs['MVTS_PARAMETERS']
        self.metadata_tags: list = configs['META_DATA_TAGS']
        self.df_all_features = pd.DataFrame()

    def do_extraction_in_parallel(self, n_jobs: int, params_name: list = None,
                                  params_index: list = None, features_name: list = None,
                                  features_index: list = None, first_k: int = None,
                                  need_interp: bool = True):
        """
        This method calls `do_extraction` in parallel (using `multiprocessing` library) with
        `n_jobs` processes.

        For more info about this method and each of its arguments, see documentation of
        `do_extraction`.

        :param n_jobs: the number of processes to be employed. This number will be used to partition
                       the dataset in a way that each process gets approximately the same number
                       of files to extract features from.
        :param params_name:
        :param params_index:
        :param features_name:
        :param features_index:
        :param first_k:
        :param need_interp:
        :return: None
        """
        import multiprocessing as mp
        # ------------------------------------------------------------
        # Collect all files (only the absolute paths)
        # ------------------------------------------------------------
        dirpath, _, all_csv = next(walk(self.path_to_root))

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
        extracted_features = manager.list()
        jobs = []
        for partition in partitions:
            process = mp.Process(target=unwrap_self_do_extraction,
                                 kwargs=({'self': self,
                                          'params_name': params_name,
                                          'params_index': params_index,
                                          'features_name': features_name,
                                          'features_index': features_index,
                                          'first_k': first_k,
                                          'need_interp': need_interp,
                                          'partition': partition,
                                          'proc_id': proc_id,
                                          'output_list': extracted_features}))

            jobs.append(process)
            proc_id = proc_id + 1

        for j in jobs:
            j.start()

        for j in jobs:
            j.join()

        self.df_all_features = pd.concat(extracted_features)
        print('\n\n\t\tAll {} processes have finished their tasks.'.format([n_jobs]))

    def do_extraction(self, params_name: list = None, params_index: list = None,
                      features_name: list = None, features_index: list = None,
                      first_k: int = None, need_interp: bool = True,
                      partition: list = None, proc_id: int = None,
                      output_list: list = None):  # TODO; last arg is actually a `ListProxy`
        """
        Computes (based on the meta data loaded in the constructor) all of the statistical
        features on the mvts data (per time series; column-wise) and stores the results in the
        public class field `df_all_features`.

        Note that only if the configuration file passed to the class constructor contains a list
        of the desired parameters and features the optional arguments can be skipped. So,
        please keep in mind the followings:

            * For parameters: a selected list of parameters (i.e., column names in mvts data)
            must be provided either through the configuration file or the method
            argument `params_name`. Also the argument `params_index` can be used to work with a
            smaller list of parameters if a list of parameters is already provided in the
            configuration file.
            * For features: A selected list of parameters (i.e., statistical features available in
            `features.feature_collection.py`) MUST be provided, as mentioned above.

        :param params_name: (Optional) A list of column names, that can be used instead of
                                the list `STATISTICAL_FEATURES` that is read from the
                                configuration file in the constructor.
        :param params_index: (Optional) A list of column indices, that can be used instead
                                  of the list `STATISTICAL_FEATURES` that is read from the
                                  configuration file in the constructor. The numbers in this list
                                  (instead of strings in `STATISTICAL_FEATURES`) can be used to
                                  confine the feature extraction to a subset of time series (
                                  columns) in the mvts data.
        :param features_name: (Optional) A list of statistical features to be calculated on all
                              time series of each mvts file. The statistical features are the
                              function names present in `features.feature_collection.py'.
        :param features_index: (Optional) A list of indices corresponding to the features
                               provided in the configuration file.
        :param first_k: (Optional) If provided, only the fist `first_k` mvts files will be
                        processed. This is mainly for getting some preliminary results in case the
                        number of mvts files is too large.
        :param need_interp: True if a linear interpolation is needed to alter the missing numerical
                            values. This only takes care of the missing values and will not
                            affect the existing ones. Set it to False otherwise. Default is True.
        :param partition: (only for internal use)
        :param proc_id: (only for internal use)
        :param output_list: (only for internal use)
        :return: None
        """
        is_parallel = False
        if proc_id is not None and output_list is not None:
            is_parallel = True
        # -----------------------------------------
        # Verify arguments
        # -----------------------------------------
        _evaluate_args(params_name, params_index, features_name, features_index)

        # -----------------------------------------
        # If features are provided using one of the optional arguments
        # override self.statistical_features with the given list.
        # -----------------------------------------
        if features_name is not None:
            self.statistical_features = features_name
        elif features_index is not None:
            self.statistical_features = [self.statistical_features[i] for i in features_index]

        # -----------------------------------------
        # Get all files (or the first first_k ones).
        # -----------------------------------------
        all_csv_files = []
        if is_parallel:
            # Use the given `partition` instead of all csv files.
            all_csv_files = partition

        else:
            _, _, all_csv_files = next(walk(self.path_to_root))
            if first_k is not None:
                # Note: If `fist_k` was used in parallel version, it was already taken into account
                # in `do_execution_in_parallel`. No need to do it again.
                all_csv_files = all_csv_files[:first_k]
        # -----------------------------------------
        # If params are provided using one of the optional arguments,
        # override self.mvts_parameters with the given list.
        # -----------------------------------------
        if params_name is not None:
            self.mvts_parameters = params_name
        elif params_index is not None:
            all_params = list(pd.read_csv(path.join(self.path_to_root, all_csv_files[0]), sep='\t'))
            self.mvts_parameters = [all_params[i] for i in params_index]

        n_features = len(self.statistical_features)
        n = len(all_csv_files)
        p_parameters = len(self.mvts_parameters)
        t_tags = len(self.metadata_tags)

        if is_parallel:
            print('\n\n\t-------------PID--{}---------------'.format(proc_id))
        else:
            print('\n\n\t-----------------------------------'.format())

        print('\t\tTotal No. of time series:\t{}'.format(n))
        print('\t\tTotal No. of Parameters:\t\t{}'.format(p_parameters))
        print('\t\tTotal No. of Features:\t\t{}'.format(n_features))
        print('\t\tTotal No. of Metadata Pieces:\t\t{}'.format(t_tags))
        print('\t\tOutput Dimensionality (N:{} X (F:{} X P:{} + T:{})):\t{}'
              .format(n, n_features, p_parameters, t_tags,
                      n * (n_features * p_parameters + t_tags)))
        print('\t-----------------------------------\n'.format())

        i = 1
        # -----------------------------------------
        # Loop through each csv file and extract its features
        # -----------------------------------------
        for f in all_csv_files:
            if not f.endswith('.csv'):
                continue

            if is_parallel:
                print('\t PID:{} --> Total Processed: {} / {}'.format(proc_id, i, n))
            else:
                console_str = '\t >>> Total Processed: {0} / {1} <<<'.format(i, n)
                sys.stdout.write("\r" + console_str)
                sys.stdout.flush()

            abs_path = path.join(self.path_to_root, f)
            df_mvts: pd.DataFrame = pd.read_csv(abs_path, sep='\t')

            # -----------------------------------------
            # Keep the requested time series of mvts only.
            # -----------------------------------------
            df_raw = pd.DataFrame(df_mvts[self.mvts_parameters], dtype=float)

            # -----------------------------------------
            # Interpolate to get rid of the NaN values.
            # -----------------------------------------
            if need_interp:
                df_raw = utils.interpolate_missing_vals(df_raw)

            # -----------------------------------------
            # Extract all the features from each column of mvts.
            # -----------------------------------------
            callable_features = extractor_utils.get_methods_for_names(self.statistical_features)
            extracted_features_df = extractor_utils.calculate_one_mvts(df_raw, callable_features)

            # -----------------------------------------
            # Extract the given meta data from this mvts name.
            # -----------------------------------------
            tags_dict = dict()
            for tag in self.metadata_tags:
                tags_dict.update({tag: utils.extract_tagged_info(f, tag)})

            # -----------------------------------------
            # Flatten the resultant dataframe and add the metadata. Suppose in the meta data,
            # some pieces of information such as id, class label, start time and end time are
            # provided. The row_df will then have these columns:
            #   ID | LAB | ST | ET | FEATURE_1 | ... | FEATURE_n
            # -----------------------------------------
            row_dfs = []
            for tag, extracted_info in tags_dict.items():
                row_dfs.append(pd.DataFrame({tag: [extracted_info]}))

            features_df = extractor_utils.flatten_to_row_df(extracted_features_df)
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
            # LOOP ENDS HERE

        if is_parallel:
            print('\n\t^^^^^^^^^^^^^^^^^^^^PID: {0}^^^^^^^^^^^^^^^^^^^^^'.format(proc_id))
        print('\n\t{0} files have been processed.'.format(i - 1))
        print('\tAs a result, a dataframe of dimension {} X {} is created.'.
              format(self.df_all_features.shape[0], self.df_all_features.shape[1]))

        if is_parallel:
            output_list.append(self.df_all_features)

    def store_extracted_features(self, output_filename):
        """
        stores the dataframe of extracted features, calculated in the method `do_extraction`,
        as a csv file. The output path is read from the configuration file, while the file name
        given as the argument here will be used as the file name.

        If the output directory given in the configuration file does not exist, it will be created
        recursively.

        :param output_filename: the name of the output csv file as the calculated data frame. If
               the '.csv' extension is not provided, it will ba appended to the given name.
        """
        # -----------------------------------------
        # Store the csv of all features to 'path_to_dest'.
        # -----------------------------------------
        if not os.path.exists(self.path_to_output):
            os.makedirs(self.path_to_output)

        if not output_filename.endswith('.csv'):
            '{}.csv'.format(output_filename)

        fname = os.path.join(self.path_to_output, output_filename)
        self.df_all_features.to_csv(fname, sep='\t', header=True, index=False)
        print('\n\tThe dataframe is stored at: {0}'.format(fname))


def main():
    path_to_config = os.path.join(CONST.ROOT, CONST.PATH_TO_CONFIG)
    fe = FeatureExtractor(path_to_config)
    # fe.do_extraction(features_name=['get_min', 'get_max', 'get_median', 'get_mean'],
    #                  params_name=['TOTUSJH', 'TOTBSQ', 'TOTPOT'], first_k=50)

    fe.do_extraction_in_parallel(n_jobs=4,
                                 features_name=['get_min', 'get_max', 'get_median', 'get_mean'],
                                 params_name=['TOTUSJH', 'TOTBSQ', 'TOTPOT'], first_k=50)

    print(fe.df_all_features.shape)
    fe.store_extracted_features('extracted_features_parallel_3_pararams_4_features.csv')


if __name__ == '__main__':
    main()
