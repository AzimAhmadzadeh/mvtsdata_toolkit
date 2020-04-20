import unittest
from .configs import config_reader_tests
from .data_analysis import mvts_data_analysis_tests
from .data_analysis import extracted_features_analysis_tests
from .features import extractor_utils_tests
from .features import feature_collection_tests
from .features import feature_extractor_tests
from .normalizing import normalizer_tests
from .sampling import sampler_tests
from .utils import mvts_cleaner_tests

# This script is created so that all unittests can be executed all at once.
#
# All unittests in all test-modules within a test-package will form a single
# TestSuite, and at the end, all TestSuites will be combined into one large
# TestSuite, called 'all_test_suites'.
#
# To run this, do:
# > python -m tests.test_runner
#
if __name__ == "__main__":

    loader = unittest.TestLoader()
    # -------------------------------------------------------
    #                  Initialize TestSuites
    # -------------------------------------------------------
    suite_config = unittest.TestSuite()
    suite_data_analysis = unittest.TestSuite()
    suite_features = unittest.TestSuite()
    suite_normalizing = unittest.TestSuite()
    suite_sampling = unittest.TestSuite()
    suite_utils = unittest.TestSuite()

    # -------------------------------------------------------
    #                  Add TestCases to TestSuites
    # -------------------------------------------------------

    # ----------- Test-suite for 'configs' ------------------
    suite_config.addTests(loader.loadTestsFromModule(config_reader_tests))

    # ----------- Test-suite for 'data_analysis' ------------
    suite_data_analysis.addTests(loader.loadTestsFromModule(mvts_data_analysis_tests))
    suite_data_analysis.addTests(loader.loadTestsFromModule(extracted_features_analysis_tests))

    # ----------- Test-suite for 'features' -----------------
    suite_features.addTests(loader.loadTestsFromModule(extractor_utils_tests))
    suite_features.addTests(loader.loadTestsFromModule(feature_collection_tests))
    suite_features.addTests(loader.loadTestsFromModule(feature_extractor_tests))

    # ----------- Test-suite for 'normalizing' --------------
    suite_normalizing.addTests(loader.loadTestsFromModule(normalizer_tests))

    # ----------- Test-suite for 'sampling' -----------------
    suite_sampling.addTests(loader.loadTestsFromModule(sampler_tests))

    # ----------- Test-suite for 'utils' --------------------
    suite_utils.addTests(loader.loadTestsFromModule(mvts_cleaner_tests))

    # -------------------------------------------------------
    #                  Combine all TestSuites into one
    # -------------------------------------------------------
    all_test_suites = unittest.TestSuite([suite_config, suite_data_analysis, suite_features,
                                         suite_normalizing, suite_sampling, suite_utils])

    runner = unittest.TextTestRunner(verbosity=3)
    result = runner.run(all_test_suites)
