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
                      need_interp: bool = True):
        """

        :param proc_id:
        :param all_csvs:
        :param output_list:
        :param params_name:
        :param params_index:
        :param features_name:
        :param features_index:
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
        # if first_k is not None:
        #     all_csv_files = all_csv_files[:first_k]

def main():
    pass


if __name__ == '__main__':
    main()

