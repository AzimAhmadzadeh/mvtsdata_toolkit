import os
import pandas as pd
from os import path, walk
import utils
import yaml
import CONSTANTS as CONST
from features import extractor_utils


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


class FeatureExtractorParallel:

    def __init__(self, path_to_config: str):
        """

        :param path_to_config:
        """
        with open(path_to_config) as file:
            configs = yaml.load(file, Loader=yaml.FullLoader)

        self.path_to_root = os.path.join(CONST.ROOT, configs['PATH_TO_MVTS'])
        self.path_to_output = os.path.join(CONST.ROOT, configs['PATH_TO_EXTRACTED_FEATURES'])
        self.statistical_features: list = configs['STATISTICAL_FEATURES']
        self.mvts_parameters: list = configs['MVTS_PARAMETERS']
        self.metadata_tags: list = configs['META_DATA_TAGS']
        self.df_all_features = pd.DataFrame()

    def calculate_all(self, proc_id: int, all_csvs: list, output_list: list,
                      params_name: list = None, params_index: list = None,
                      features_name: list = None, features_index: list = None,
                      first_k: int = None, need_interp: bool = True):
        """

        :param proc_id:
        :param all_csvs:
        :param output_list:
        :param params_name:
        :param params_index:
        :param features_name:
        :param features_index:
        :param first_k:
        :param need_interp:
        :return:
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

        # -----------------------------------------
        # If params are provided using one of the optional arguments,
        # override self.mvts_parameters with the given list.
        # -----------------------------------------
        if params_name is not None:
            self.mvts_parameters = params_name
        elif params_index is not None:
            all_params = list(pd.read_csv(path.join(dirpath, all_csv_files[0]), sep='\t'))
            self.mvts_parameters = [all_params[i] for i in params_index]

        n_features = len(self.statistical_features)
        n = len(all_csv_files)
        p_parameters = len(self.mvts_parameters)
        t_tags = len(self.metadata_tags)
        print('\n\n\t--------------PID--{}-------------------'.format(proc_id))
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
        # Loop through each csv file and extract the features
        # -----------------------------------------
        for f in all_csvs:
            print('\t PID:{} --> Total Processed: {} / {}'.format(proc_id, i, n))

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

        print('\n\t^^^^^^^^^^^^^^^^^^^^PID: {}^^^^^^^^^^^^^^^^^^^^^'.format(proc_id))
        print('\tDone! {} files have been processed.'.format(i - 1))
        print('\tIn total, a dataframe of dimension {} X {} is created.'.format(
            self.df_all_features.shape[0],
            self.df_all_features.shape[1]))
        print('\t^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n'.format(proc_id))

        output_list.append(self.df_all_features)


def main():
    pass


if __name__ == '__main__':
    main()

