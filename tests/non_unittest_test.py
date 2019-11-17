import CONSTANTS as CONST
import pandas as pd
import os
from sampling.sampler import Sampler
from data_analysis.extracted_features_analysis import ExtractedFeaturesAnalysis

'''
This file takes care of the testing which are hard to replicate using unittest.

'''

if __name__ == "__main__":

    f_path = os.path.join(CONST.ROOT,
                          'pet_datasets/extracted_features')
    path_to_data = os.path.join(f_path, "extracted_features.csv")
    df = pd.read_csv(path_to_data, sep='\t')
    # Execute the class population method from Sampler class and match the results
    s = Sampler(df, 'lab')
    #s.undersample(['X','M'],['B','NF','C'],'X')
    s.oversample(['X','M'],['B','NF','C'],'C')
    #s.sample(desired_ratios= {'B': 1,'NF': .33, 'M': 1, 'C': -1 , 'X':.15})
    print(s.desired_dfs.shape[0])
    labels = s.get_labels()
    df_dict = s.get_decomposed_mvts()

    populations = s.get_original_populations()

    # Testing ExtractedFeaturesAnalysis class on original extracted_features.csv
    m = ExtractedFeaturesAnalysis(df)
    # compute_summary() needs to be called first in order to print/save any Data Analysis results
    m.compute_summary()
    print(m.summary)
    # m.print_summary(output_filename='output.csv')
    class_population = m.get_class_population
    print(class_population)
    miss_pop = m.get_missing_values()

    five_num = m.get_five_num_summary()
    print('Print Class Population:')
    print(class_population)
    print('Print Missing Value:')
    print(miss_pop)
    print('Print Five Num Value:')
    print(five_num)

    # Now lets create non_unittest_extracted_feature.csv file from the original extracted_feature
    # file manipulating some values in order to
    # test the methods of mvts_data_analysis

    path_to_testdata = CONST.OUT_PATH_TO_EXTRACTED_FEATURES
    path_to_testdata = os.path.join(path_to_testdata, "non_unittest_extracted_features.csv")
    df_test = pd.read_csv(path_to_testdata, sep='\t')

    # Testing ExtractedFeaturesAnalysis class on test file non_unittest_extracted_features.csv
    m = ExtractedFeaturesAnalysis(df_test)
    # compute_summary() needs to be called first in order to print or save any Data Analysis results
    m.compute_summary()
    class_population = m.get_class_population

    miss_pop = m.get_missing_values()

    five_num = m.get_five_num_summary()
    print('Print Class Population on Test Data:')
    print(class_population)
    print('Print Missing Value on Test Data:')
    print(miss_pop)
    print('Print Five Num Value on Test Data:')
    print(five_num)
