import os

ROOT = os.path.dirname(__file__)
PATH_TO_CONFIG = os.path.join(ROOT, 'feature_extraction_configs.yml')
PATH_TO_DATASETS_CONFIG = os.path.join(ROOT, 'datasets_configs.yml')
DATASETS_DICT = {1: 'https://bitbucket.org/gsudmlab/mvts_data_toolkit/downloads/petdataset_01.zip',
                 2: ''}
PATH_TO_EXTRACTED_FEATURES_TEST = \
    os.path.join(ROOT, 'tests/test_dataset/non_unittest_extracted_features.csv')

