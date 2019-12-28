import os
import yaml

_config_keys = ['PATH_TO_MVTS', 'PATH_TO_EXTRACTED_FEATURES',
                'META_DATA_TAGS', 'MVTS_PARAMETERS', 'STATISTICAL_FEATURES']


class ConfigReader:
    """
    This is a simple class to read the configuration file and meanwhile assert its content. This
    class, instead of direct reading of the configuration file, must be used whenever the
    configuration file is needed.

    Note: Should any new entry be added to the main structure of the configuration file, it must be
    considered in this class as well.
    """
    def __init__(self, path_to_config: str):
        self.path_to_config = path_to_config

    def read(self) -> dict:
        configs = None
        file_assessment = self.__assert_file()
        if file_assessment is True:
            with open(self.path_to_config) as file:
                configs = yaml.load(file, Loader=yaml.FullLoader)
        self.__assert_content(configs)
        return configs

    def __assert_file(self):
        if not os.path.isfile(self.path_to_config):
            self.__invalid_path_msg()
            return False

        if not self.path_to_config.endswith('.yml'):
            self.__invalid_file_msg()
            return False

        return True

    def __assert_content(self, configs: dict):
        if not list(configs.keys()) == _config_keys:
            self.__invalid_content_msg()
            return False

    def __invalid_path_msg(self):
        raise FileNotFoundError(
            '''
            The given configuration file does NOT exists:
            \t{}
            A configuration file must be provided. For help, call `instruction()`.
            '''.format(self.path_to_config))

    def __invalid_file_msg(self):
        raise FileNotFoundError(
            '''
            The given configuration file it NOT a YAML file:
            \t{}
            A configuration file must be a YAML file. For help, call `instruction()`.
            '''.format(self.path_to_config)
        )

    def __invalid_content_msg(self):
        print('Some of the keys are invalid or missing in the given configuration file:\n'
              '\t{}'.format(self.path_to_config))
        raise AssertionError(
            '''
            The keys in the following configuration file does NOT match
            with the keys in a valid configuration file.
            {}
            For help, 
            call `instruction()`.'
            '''.format(self.path_to_config)
            )

    def instruction(self):
        print("""
        The content of the configuration file MUST have all of the following items:
        ---------------------------------------------------------------------------
            PATH_TO_MVTS: '/relative/path/to/data/'
            PATH_TO_EXTRACTED_FEATURES: '/relative/path/to/extracted_features/'
            META_DATA_TAGS: ['id', 'lab']
            MVTS_PARAMETERS:
                - 'param 1'
                - 'param 2'
                - 'param 3'
            STATISTICAL_FEATURES:
                - 'get_min'
                - 'get_max'
        ---------------------------------------------------------------------------
        The above example, contains 3 parameters of the multivariate time series
        dataset (i.e., 'param 1', 'param 2', and 'param 3', and 2 of the features from 
        `features.features_collection` module (i.e., 'get_min' and 'get_max').
        
        Note 1: The values under MVTS_PARAMETERS must be identical to those used in
        the dataset as the column-names. Users may want to list only those parameters
        which are needed. Those that are not mentioned, will be ignored in processes.
        
        Note 2: PATH_TO_EXTRACTED_FEATURES points to where the extracted features
        will be stored after they are computed using `features.feature_extractor`
        module. Later on, when other modules need to access the extracted features,
        the program knows, using this value, where it is located.        
        
        Note 3: The values under STATISTICAL_FEATURES are in fact the names of the
        methods provided in `features.feature_collection` module. The names should
        be identical to those methods, without the parenthesis. Users may want to
        pick those that are interesting and meaningful for their time series.
        """)


if __name__ == "__main__":
    import CONSTANTS as CONST
    path_to_config = os.path.join(CONST.ROOT, CONST.PATH_TO_CONFIG)
    # path_to_config = path_to_config[:-4] + '.xml'
    cr = ConfigReader(path_to_config)
    # cr.instruction()
    conf = cr.read()
    print(conf)
