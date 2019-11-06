import pandas as pd
import os
import sys
from os import path, walk
import utils
from features import extractor_utils
import yaml
import CONSTANTS as CONST


class FeatureExtractor:
    """
    This class loads the configuration file 'feature_extraction_configs.yml', and read the
    following pieces of information from it:
     - PATH_TO_MVTS: path to where the csv (mvts) files are stored.
     - MVTS_PARAMETERS: a list of time series name; only those listed here will be processed.
     - STATISTICAL_FEATURES: a list of statistical features to be extracted from each time series.
     - META_DATA_TAGS: a list of tags used in the mvts file names; to be used for extraction of
     some metadata from file names.
     - PATH_TO_EXTRACTED_FEATURES: path to a directory where the extracted features (one csv
     file) will be stored.

    Based on these values, it walks through the directory PATH_TO_MVTS and for each of the mvts
    files, it computes the statistical features STATISTICAL_FEATURES on all time series listed in
    MVTS_PARAMETERS. It used the tags in META_DATA_TAGS to extract some metadata, such as class
    label, time stamp, id, etc.

    The resultant dataframe (i.e., the extracted features) will have T X F + x columns, where F is
    the total number of features (in , STATISTICAL_FEATURES), T is the total number of
    time series parameters (in MVTS_PARAMETERS), and x is the number of meta data extracted from
    the file names (i.e., number of tags in META_DATA_TAGS). In the extracted features dataframe,
    the column-name of the nominal attributes is of the structure
    <TIME_SERIES_NAME>_<statistic_name>. For instance, for a time series named 'DENSITY' and the
    statistical feature 'mean', the corresponding column-name would be 'DENSITY_mean'.
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
        self.statistical_features: list = \
            extractor_utils.get_methods_for_names(configs['STATISTICAL_FEATURES'])
        self.mvts_parameters: list = configs['MVTS_PARAMETERS']
        self.metadata_tags: list = configs['META_DATA_TAGS']
        self.df_all_features = pd.DataFrame()

    def calculate_all(self, params_index_list: list = None, need_interp: bool = True):
        """
        Computes (based on the meta data loaded in the constructor) all of the statistical
        features on the mvts data (per time series; column-wise) and stores the results in the
        class-field `df_all_features`.

        :param params_index_list: A list of column indices, that can be used instead of the list
        STATISTICAL_FEATURES that is read from the configuration file in the constructor. The
        numbers in this list (instead of strings in STATISTICAL_FEATUES) can be used to confine
        the feature extraction to a subset of time series (columns) in the mvts data.
        :param need_interp: True if a linear interpolation is needed to alter the missing numerical
        values. This only takes care of the missing values and will not affect the existing ones.
        Set it to False otherwise. Default is True.
        """
        # -----------------------------------------
        # Verify arguments
        # -----------------------------------------
        has_param_name_arg = (self.mvts_parameters is not None) and (len(self.mvts_parameters) > 0)
        has_param_index_arg = (params_index_list is not None) and (len(params_index_list) > 0)
        if has_param_name_arg == has_param_index_arg:  # mutual exclusive
            raise ValueError(
                """
                One and only one of the two arguments (params_name_list, params_index_list) must
                be provided.
                """
            )

        if len(self.statistical_features) == 0:
            raise ValueError(
                """
                The argument 'self.statistical_features' cannot be empty!
                """
            )
        # -----------------------------------------
        # Get all file names in the root directory
        # -----------------------------------------
        print(self.path_to_root)
        dirpath, _, all_csv_files = next(walk(self.path_to_root))
        n = len(all_csv_files)
        n_features = len(self.statistical_features)
        i = 1

        print('\n\n\t-----------------------------------'.format())
        print('\t\tTotal No. of Features:\t\t{}'.format(n_features))
        print('\t\tTotal No. of time series:\t{}'.format(n))
        print('\t\tOutput TS dimensionality ({} X {}):\t{}'.format(n, n_features, n * n_features))
        print('\t-----------------------------------\n'.format())

        # -----------------------------------------
        # Loop through each csv file and extract the features
        # -----------------------------------------
        for f in all_csv_files:
            if i > 10:
                break
            if f.lower().find('.csv') != -1:
                print('\t >>> Total Processed: {0} / {1} <<<\r'.format(i, n))
                sys.stdout.flush()

                abs_path = path.join(dirpath, f)
                df_mvts: pd.DataFrame = pd.read_csv(abs_path, sep='\t')

                # -----------------------------------------
                # Keep the requested time series of mvts only.
                # -----------------------------------------
                df_raw = pd.DataFrame()
                if has_param_name_arg:
                    df_raw = pd.DataFrame(df_mvts[self.mvts_parameters], dtype=float)
                elif has_param_index_arg:
                    df_raw = pd.DataFrame(df_mvts.iloc[:, params_index_list], dtype=float)

                # -----------------------------------------
                # Interpolate to get rid of the NaN values.
                # -----------------------------------------
                if need_interp:
                    df_raw = utils.interpolate_missing_vals(df_raw)

                # -----------------------------------------
                # Extract all the features from each column of mvts.
                # -----------------------------------------
                extracted_features_df =\
                    extractor_utils.calculate_one_mvts(df_raw, self.statistical_features)

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
    pc.calculate_all()
    pc.store_extracted_features('extracted_features_tmp.csv')


if __name__ == '__main__':
    main()
