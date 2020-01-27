import pandas as pd
import os
import sys
from os import path, walk
import utils
from features import extractor_utils
import CONSTANTS as CONST
from configs.config_reader import ConfigReader


def _evaluate_features(features_name: list, features_index: list, config_features_available: bool):
    """
    This ensures that (1) if feature names are provided in the config file, one and only one of the
    arguments, `features_name` or `features_index`, might be given, and (2) if feature names are
    NOT provided in the config file, one and only one of the arguments MUST be given.
    :param features_name: See the same argument in `do_extraction`.
    :param features_index: See the same argument in `do_extraction`.
    :param config_features_available: `True` if the key `STATISTICAL_FEATURES` in the config file,
           is associated with a list of features, and `False` otherwise.
    :return: True, if no exception was raised.
    """
    if features_index is None: features_index = []
    if features_name is None: features_name = []
    given_by_list, given_by_index = False, False
    if len(features_name) > 0:
        given_by_list = True
    if len(features_index) > 0:
        given_by_index = True

    if not config_features_available:  # if features in config file are not provided
        if given_by_list and not given_by_index:
            return True
        else:
            # if (1) both args are given, or (2) if none of them are provided, or (3) if
            # params_index is given
            raise ValueError(
                """
                If a list of feature names is not provided by the config file, 
                the arg `features_name` and only that MUST be given.
                """
            )
    else:
        if given_by_list + given_by_index > 1:  # if both args are provided
            raise ValueError(
                """
                Both of the arguments, `features_name` and `features_index`, cannot be given at the 
                same time.
                """
            )
    return True


def _evaluate_params(params_name: list, params_index: list, config_params_available: bool):
    """
    This ensures that (1) if parameter names are provided in the config file, one and only one of
    the arguments, `params_name` or `params_index`, might be given, and (2) if parameter names are
    NOT provided in the config file, the arg `params_name` and only that MUST be given.
    :param params_name: See the same argument in `do_extraction`.
    :param params_index: See the same argument in `do_extraction`.
    :param config_params_available: `True` if the key `MVTS_PARAMETERS` in the config file, is
           associated with a list of parameters, and `False` otherwise.
    :return: True, if no exception was raised.
    """
    if params_index is None: params_index = []
    if params_name is None: params_name = []

    given_by_list, given_by_index = False, False
    if len(params_name) > 0:
        given_by_list = True
    if len(params_index) > 0:
        given_by_index = True

    if not config_params_available:  # if parameters in config file are not provided
        if given_by_list and not given_by_index:
            return True
        else:
            # if (1) both args are given, or (2) if none of them are provided, or (3) if
            # params_index is given
            raise ValueError(
                """
                If a list of parameter names is not provided by the config file, 
                the arg `params_name` and only that MUST be given.
                """
            )
    else:
        if given_by_list + given_by_index > 1:  # if both args are provided
            raise ValueError(
                """
                Both of the arguments, `params_name` and `params_index`, cannot be given at the 
                same time.
                """
            )
    return True


def _unwrap_self_do_extraction(*arg, **kwarg):
    """
    This unwraps a class method so that it can perform independently and thus can be called in
    parallel. More specifically, we need to be able to call `do_extraction` both sequentially and
    in parallel. In case of a parallel call, when it is called in a child process, the child process
    gets its own separate copy of the class instance. So, the class method needs to be unwrapped
    to a non-class method so that the class variables don't overlap.
    :param arg:
    :param kwarg:
    :return:
    """
    return FeatureExtractor.do_extraction(*arg, **kwarg)


class FeatureExtractor:
    """
    An instance of this class can extract a set of given statistical features from a large number of
    MVTS data, in both sequential and parallel fashions. It loads the configuration file
    provided by the user and reads the following pieces of information from it.

    Below are the column names of the summary dataframe:
        * `PATH_TO_MVTS`: path to where the CSV (MVTS) files are stored.
        * `MVTS_PARAMETERS`: a list of time series name; only those listed here will be processed.
        * `STATISTICAL_FEATURES`: a list of statistical features to be computed on each time series.
        * `META_DATA_TAGS`: a list of tags used in the MVTS file names; to be used for extraction of
        some metadata from file names.
        * `PATH_TO_EXTRACTED_FEATURES`: path to a directory where the extracted features (one CSV
        file) will be stored.

    Based on these values, it walks through the directory `PATH_TO_MVTS` and for each of the MVTS
    files, it computes the statistical features listed in `STATISTICAL_FEATURES` on all time series
    listed in `MVTS_PARAMETERS`. It uses the tags in `META_DATA_TAGS` to extract some metadata,
    such as class `label`, `time stamp`, `id`, etc.

    The resultant dataframe (i.e., the extracted features) will have `T X F + x` columns, where `F`
    is the total number of features (i.e., `len(STATISTICAL_FEATURES)`), `T` is the total number of
    time series parameters (i.e., `len(MVTS_PARAMETERS)`), and `x` is the number of metadata
    extracted from the file names (i.e., `len(META_DATA_TAGS)`).

    In the extracted features dataframe, the column-name of the nominal attributes is of the
    following structure::

        <TIME_SERIES_NAME>_<statistic_name>

    For instance, for a time series named `DENSITY` and the statistical feature `mean`, the
    corresponding column-name would be `DENSITY_mean`.

    Note: In `do_extraction_in_parallel`, each child process takes a list of file names (not the
    actual files) that is a partition of the entire dataset, and works independently on the MVTS
    in that partition. Therefore, the memory consumption of using `n` child processes is almost
    equal to n times the amount used in the sequential mode. That is, the parallel mode does not
    increase memory consumption exponentially with respect to the number of children. The number
    of partitions is equal to the number of child processes (i.e., `n_jobs`).
    """

    def __init__(self, path_to_config: str):
        """
        This constructor loads all necessary information from the config file located at
        `path_to_config`.

        :param path_to_config: Path to where the corresponding configuration (YAML) file is located.
        """
        cr = ConfigReader(path_to_config)
        configs = cr.read()

        self.path_to_root = os.path.join(CONST.ROOT, configs['PATH_TO_MVTS'])
        self.path_to_output = os.path.join(CONST.ROOT, configs['PATH_TO_EXTRACTED_FEATURES'])
        self.statistical_features: list = configs['STATISTICAL_FEATURES']
        self.mvts_parameters: list = configs['MVTS_PARAMETERS']
        self.metadata_tags: list = configs['META_DATA_TAGS']
        self.df_all_features = pd.DataFrame()

    def do_extraction_in_parallel(self, n_jobs: int, params_name: list = None,
                                  params_index: list = None, features_name: list = None,
                                  features_index: list = None, first_k: int = None,
                                  need_interp: bool = True, verbose: bool = False):
        """
        This method calls `do_extraction` in parallel (using `multiprocessing` library) with
        `n_jobs` processes.

        For more info about this method and each of its arguments, see documentation of
        `do_extraction`.

        :param n_jobs: The number of processes to be employed. This number will be used to partition
                       the dataset in a way that each process gets approximately the same number
                       of files to extract features from.
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
            process = mp.Process(target=_unwrap_self_do_extraction,
                                 kwargs=({'self': self,
                                          'params_name': params_name,
                                          'params_index': params_index,
                                          'features_name': features_name,
                                          'features_index': features_index,
                                          'first_k': first_k,
                                          'need_interp': need_interp,
                                          'partition': partition,
                                          'proc_id': proc_id,
                                          'verbose': verbose,
                                          'output_list': extracted_features}))

            jobs.append(process)
            proc_id = proc_id + 1

        for j in jobs:
            j.start()

        for j in jobs:
            j.join()

        self.df_all_features = pd.concat(extracted_features)
        self.df_all_features.reset_index(drop=True, inplace=True)
        if verbose:
            print('\n\n\t\tAll {} processes have finished their tasks.'.format([n_jobs]))

    def do_extraction(self, params_name: list = None, params_index: list = None,
                      features_name: list = None, features_index: list = None,
                      first_k: int = None, need_interp: bool = True,
                      partition: list = None, proc_id: int = None, verbose: bool = False,
                      output_list: list = None):
        """
        Computes (based on the metadata loaded in the constructor) all of the statistical
        features on the MVTS data (per time series; column-wise) and stores the results in the
        public class field `df_all_features`.

        Note that only if the configuration file passed to the class constructor contains a list
        of the desired parameters and features the optional arguments can be skipped. So,
        please keep in mind the followings:

            * For parameters: a selected list of parameters (i.e., column names in MVTS data) must
              be provided either through the configuration file or the method argument
              `params_name`. Also, the argument `params_index` can be used to work with a smaller
              list of parameters if a list of parameters is already provided in the config file.
            * For features: A selected list of parameters (i.e., statistical features available
              in `features.feature_collection.py`) MUST be provided, as mentioned above.

        :param params_name: (Optional) A list of parameter names of interest that can be used
                            instead of the list `MVTS_PARAMETERS` given in the config file. If
                            the list in the config file is NOT provided, then either this or
                            `params_index` MIST be given.
        :param params_index: (Optional) A list of column indices of interest that can be used
                             instead of the list `MVTS_PARAMETERS` given in the config file.
                             If the list in the config file is NOT provided, then either this or
                             `params_name` MUST be given.
        :param features_name: (Optional) A list of statistical features to be calculated on all
                              time series of each MVTS file. The statistical features are the
                              function names present in `features.feature_collection.py'. If they
                              are not provided in the config file (under `STATISTICAL_FEATURES`),
                              either this or `features_index` MUST be given.
        :param features_index: (Optional) A list of indices corresponding to the features
                               provided in the configuration file. If they are not provided in
                               the config file (under `STATISTICAL_FEATURES`), either this or
                               `features_names` MUST be given.
        :param first_k: (Optional) If provided, only the fist `first_k` MVTS files will be
                        processed. This is mainly for getting some preliminary results in case the
                        number of MVTS files is too large.
        :param need_interp: True if a linear interpolation is needed to alter the missing numerical
                            values. This only takes care of the missing values and will not
                            affect the existing ones. Set it to False otherwise. Default is True.
        :param partition: (only for internal use)
        :param proc_id: (only for internal use)
        :param verbose: If set to True, the program prints on the console which files are being
                        processed and what processes (if parallel) are doing the work. The default
                        value is False.
        :param output_list: (only for internal use)
        :return: None
        """
        is_parallel = False
        if proc_id is not None and output_list is not None:
            is_parallel = True
        # -----------------------------------------
        # Verify arguments
        # -----------------------------------------
        _evaluate_params(params_name, params_index,
                         config_params_available=self.mvts_parameters is not None)
        _evaluate_features(features_name, features_index,
                           config_features_available=self.statistical_features is not None)
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
                # Note: If `fist_k` was used in parallel version, it should have already been taken
                # into account in `do_execution_in_parallel`. So, no need to do it again.
                all_csv_files = all_csv_files[:first_k]
        # -----------------------------------------
        # If params are provided using one of the optional arguments,
        # override self.mvts_parameters with the given list.
        # -----------------------------------------
        if params_name is not None:
            self.mvts_parameters = params_name
        elif params_index is not None:
            self.mvts_parameters = [self.mvts_parameters[i] for i in params_index]

        n_features = len(self.statistical_features)
        n = len(all_csv_files)
        p_parameters = len(self.mvts_parameters)
        t_tags = len(self.metadata_tags)

        if verbose:
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

            if verbose:
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

        self.df_all_features.reset_index(drop=True, inplace=True)

        if verbose:
            if is_parallel:
                print('\n\t^^^^^^^^^^^^^^^^^^^^PID: {0}^^^^^^^^^^^^^^^^^^^^^'.format(proc_id))
            print('\n\t{0} files have been processed.'.format(i - 1))
            print('\tAs a result, a dataframe of dimension {} X {} is created.'.
                  format(self.df_all_features.shape[0], self.df_all_features.shape[1]))

        if is_parallel:
            output_list.append(self.df_all_features)

    def store_extracted_features(self, output_filename:str, verbose:bool=True):
        """
        Stores the dataframe of extracted features, calculated in the method `do_extraction`,
        as a CSV file. The output path is read from the configuration file, while the file name
        given as the argument here will be used as the file name.

        If the output directory given in the configuration file does not exist, it will be created
        recursively.

        :param output_filename: The name of the output CSV file as the calculated data frame. If
               the '.csv' extension is not provided, it will ba appended to the given name.
        :param verbose: Set to `False` to prevent the output path be printed on console. Default
                        is set to True.
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
        if verbose:
            print('\n\tThe dataframe is stored at: {0}'.format(fname))

    def plot_boxplot(self, feature_names: list, output_path: str = None):
        """
        Generates a plot of box-plots, one for each extracted feature.

        :param feature_names: A list of feature-names indicating the columns of interest for this
                              visualization.
        :param output_path: If given, the generated plot will be stored instead of shown.
                            Otherwise, it will be only shown if the running environment allows it.
        :return: None
        """
        from visualizations.stat_visualizer import StatVisualizer
        sv = StatVisualizer(extracted_features=self.df_all_features)
        sv.boxplot_extracted_features(feature_names=feature_names, output_path=output_path)

    def plot_violinplot(self, feature_names: list, output_path: str = None):
        """
        Generates a plot of violin-plots, one for each extracted feature.

        :param feature_names: A list of feature-names indicating the columns of interest for this
                              visualization.
        :param output_path: If given, the generated plot will be stored instead of shown.
                            Otherwise, it will be only shown if the running environment allows it.
        :return: None
        """
        from visualizations.stat_visualizer import StatVisualizer
        sv = StatVisualizer(extracted_features=self.df_all_features)
        sv.plot_violinplot(feature_names=feature_names, output_path=output_path)

    def plot_splom(self, feature_names: list, output_path: str = None):
        """
        Generates a SPLOM, or a scatter plot matrix, for all pairs of features. Note that for a
        large number of features this may take a while (since each cell of the matrix is a
        scatter plot on its own), and also the final plot may become very large.

        :param feature_names: A list of feature-names indicating the columns of interest for this
                              visualization.
        :param output_path: If given, the generated plot will be stored instead of shown.
                            Otherwise, it will be only shown if the running environment allows it.
        :return: None
        """
        from visualizations.stat_visualizer import StatVisualizer
        sv = StatVisualizer(extracted_features=self.df_all_features)
        sv.plot_splom(feature_names=feature_names, output_path=output_path)

    def plot_correlation_heatmap(self, feature_names: list, output_path: str = None):
        """
        Generates a heat-map for the correlation matrix of all pairs of given features.

        Note: Regardless of the range of correlations, the color-map is fixed to [-1, 1]. This is
        especially important to avoid mapping insignificant changes of values into significant
        changes of colors.

        :param feature_names: A list of feature-names indicating the columns of interest for this
                              visualization.
        :param output_path: If given, the generated plot will be stored instead of shown.
                            Otherwise, it will be only shown if the running environment allows it.
        :return: None
        """
        from visualizations.stat_visualizer import StatVisualizer
        sv = StatVisualizer(extracted_features=self.df_all_features)
        sv.plot_correlation_heatmap(feature_names=feature_names, output_path=output_path)

    def plot_covariance_heatmap(self, feature_names: list, output_path: str = None):
        """
        Generates a heat-map for the covariance matrix of all pairs of given features.

        Note that covariance is not a standardized statistic, and because of this, the color-map
        might be confusing; when the difference between the largest and smallest covariance is
        insignificant, the colors may imply a significant difference. To avoid this, the values
        mapped to the colors (as shown next to the color-map) must be carefully taken into
        account in the analysis of the covariance.

        :param feature_names: A list of feature-names indicating the columns of interest for this
                              visualization.
        :param output_path: If given, the generated plot will be stored instead of shown.
                            Otherwise, it will be only shown if the running environment allows it.
        :return: None
        """
        from visualizations.stat_visualizer import StatVisualizer
        sv = StatVisualizer(extracted_features=self.df_all_features)
        sv.plot_covariance_heatmap(feature_names=feature_names, output_path=output_path)
