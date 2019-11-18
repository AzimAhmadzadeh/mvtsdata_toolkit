import pandas as pd
import numpy as np
import os
import sys
from os import path, walk
import utils
from features import extractor_utils
import yaml
import CONSTANTS as CONST


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
        This constructor loads all necessary information from the config file provided by the
        given path.

        :param path_to_config: path to where the corresponding configuration (YAML) file is located.
        """
        with open(path_to_config) as file:
            configs = yaml.load(file, Loader=yaml.FullLoader)

        self.path_to_root = os.path.join(CONST.ROOT, configs['PATH_TO_MVTS'])
        self.path_to_output = os.path.join(CONST.ROOT, configs['PATH_TO_EXTRACTED_FEATURES'])
        self.statistical_features: list = configs['STATISTICAL_FEATURES']
        self.mvts_parameters: list = configs['MVTS_PARAMETERS']
        self.metadata_tags: list = configs['META_DATA_TAGS']
        self.df_all_features = pd.DataFrame()

    def calculate_all(self, params_name: list = None, params_index: list = None,
                      features_name: list = None, features_index: list = None,
                      first_k: int = None, need_interp: bool = True):
        """
        Computes (based on the meta data loaded in the constructor) all of the statistical
        features on the mvts data (per time series; column-wise) and stores the results in the
        class-field `df_all_features`.

        :param params_name: (Optional) A list of column names, that can be used instead of
                                the list `STATISTICAL_FEATURES` that is read from the
                                configuration file in the constructor.
        :param params_index: (Optional) A list of column indices, that can be used instead
                                  of the list `STATISTICAL_FEATURES` that is read from the
                                  configuration file in the constructor. The numbers in this list
                                  (instead of strings in STATISTICAL_FEATUES) can be used to confine
                                  the feature extraction to a subset of time series (columns) in the
                                  mvts data.
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

        :return: None
        """
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
        # Get all files (or the first first_k ones) in the root directory.
        # -----------------------------------------
        print(self.path_to_root)
        dirpath, _, all_csv_files = next(walk(self.path_to_root))
        if first_k is not None:
            all_csv_files = all_csv_files[:first_k]

        n = len(all_csv_files)
        n_features = len(self.statistical_features)
        i = 1

        # -----------------------------------------
        # If params are provided using one of the optional arguments
        # override self.mvts_parameters with the given list.
        # -----------------------------------------
        if params_name is not None:
            self.mvts_parameters = params_name
        elif params_index is not None:
            all_params = list(pd.read_csv(path.join(dirpath, all_csv_files[0]), sep='\t'))
            self.mvts_parameters = [all_params[i] for i in params_index]

        p_parameters = len(self.mvts_parameters)
        t_tags = len(self.metadata_tags)
        print('\n\n\t-----------------------------------'.format())
        print('\t\tTotal No. of time series:\t{}'.format(n))
        print('\t\tTotal No. of Parameters:\t\t{}'.format(p_parameters))
        print('\t\tTotal No. of Features:\t\t{}'.format(n_features))
        print('\t\tTotal No. of Metadata Pieces:\t\t{}'.format(t_tags))
        print('\t\tOutput dimensionality (N:{} X (F:{} X P:{} + T:{})):\t{}'
              .format(n, n_features, p_parameters, t_tags,
                      n * (n_features * p_parameters + t_tags)))
        print('\t-----------------------------------\n'.format())

        # -----------------------------------------
        # Loop through each csv file and extract the features
        # -----------------------------------------
        for f in all_csv_files:
            if f.lower().find('.csv') != -1:
                console_str = '\t >>> Total Processed: {0} / {1} <<<'.format(i, n)
                sys.stdout.write("\r" + console_str)
                sys.stdout.flush()

                abs_path = path.join(dirpath, f)
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
                callable_features = extractor_utils.get_methods_for_names(  # <<<<<<<<<<<<<<<
                    self.statistical_features)
                extracted_features_df = \
                    extractor_utils.calculate_one_mvts(df_raw, callable_features)

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

        print('\n\t{0} files have been processed.'.format(i - 1))
        print('\tAs a result, a dataframe of dimension {} X {} is created.'.
              format(self.df_all_features.shape[0], self.df_all_features.shape[1]))

    def store_extracted_features(self, output_filename):
        """
        stores the dataframe of extracted features, calculated in the method `calculate_all`,
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
    pc = FeatureExtractor(path_to_config)
    # pc.calculate_all()
    pc.calculate_all(#features_name=['get_min', 'get_max', 'get_median', 'get_mean'],
                    features_index=[1, 4, 6],
                     params_index=[5, 6, 7], first_k=50)
    # pc.store_extracted_features('extracted_features_3_pararams_3_featues.csv')


if __name__ == '__main__':
    main()