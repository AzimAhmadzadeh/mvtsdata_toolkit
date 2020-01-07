import pandas as pd
import numpy as np
import copy
from sampling.input_validator import validate_input, validate_sampling_input


class Sampler:
    """
    This module  contains several sampling methods to handle dataset issues like imbalanced class.

    See ReadMe.md for usage example.
    """

    def __init__(self, mvts_df: pd.DataFrame, label_col_name):
        """
        The constructor method takes the input dataset as pandas dataframe and the column name to decompose
        the specific column or feature.
        We can get the population and ratio of the provided column name.
        It also decomposes the whole dataset in several smaller dataset based on class.

        :param mvts_df:
        :param label_col_name:
        """
        self.df = mvts_df
        self.label_col_name = label_col_name
        self.labels = mvts_df[label_col_name].unique().tolist()
        self.dfs_dict = {label: pd.DataFrame for label in self.labels}
        self.desired_dfs = pd.DataFrame()  # {label: pd.DataFrame for label in self.labels}
        self.__decompose_mvts()
        self.class_population = self.__compute_original_population()
        self.class_ratios = self.__compute_original_ratios()

    def __compute_original_population(self):
        """
        A private method that computes the sample population of the original dataset.
        :return: a dictionary of labels (as key) and populations (as values)
        """
        pop_dict = {label: len(self.dfs_dict[label]) for label in self.dfs_dict.keys()}
        return pop_dict

    def __compute_original_ratios(self):
        total = sum(self.class_population.values())
        class_ratios = [self.class_population[label] / total for label in
                        self.class_population.keys()]
        return class_ratios

    def __decompose_mvts(self):
        """
        A private method that decomposes the mvts dataframe into several dataframes, one for each
        class label, in the form of a dictionary.
        """
        for key in self.dfs_dict.keys():
            self.dfs_dict[key] = self.df[:][self.df[self.label_col_name] == key]

    def get_labels(self):
        """
        A getter method for the labels
        :return: a list of all class labels of the data.
        """
        return self.labels

    def get_original_populations(self):
        """
        Gets the per-class population of the original dataset.
        :return: a dictionary of labels (as key) and class populations (as values)
        """
        return self.class_population

    def get_original_ratios(self):
        """
        Gets the per-class ratio of the original dataset.
        :return: a dictionary of labels (as key) and class ratios (as values)
        """
        return self.class_ratios

    def get_decomposed_mvts(self):
        """
        Decomposes the dataframe provided to the class into several dataframes, one for each class
        label.
        :return: a dictionary of dataframes, with keys being the class labels, and the corresponding
                 dataframes contain instances of those labels only.
        """
        return self.dfs_dict

    def sample(self, desired_populations: dict = None, desired_ratios: dict = None):
        """
        Using this method one could do either undersampling or oversampling, in the most generic
        fashion. That is, the user determines the expected population size or ratios that they
        would like to get from the mvts data.
        Example: Consider a mvts data with 5 classes:

            |A| = 100, |B| = 400, |C| = 300, |D| = 700, |E| = 2000

        and given is: desired_ratios = [-1, -1, 0.33, 0.33, 0.33]

        then, the instances of classes A and B will not change, while
        the population size of each of the  C, D, and E classes would be
        one-third of the sample size (3500 * 0.33).

        Note:
            1. One and only one of the arguments must be provided.
            2. The dictionary must contain all class labels present in the mvts dataframe.
            3. The number -1 can be used wherever the population or ratio should not change.

        :param desired_populations: a dictionary of label-integer pairs, where each integer specifies
        the desired population of the corresponding class. The integers must be positive, but -1
        can be used to indicate that the population of the corresponding class should remain unchanged.
        :param desired_ratios: a dictionary of label-float pairs, where each float specifies
        the desired ratios (with respect to the total sample size) of the corresponding class.
        The floats must be positive, but -1 can be used to indicate that the ratio of the corresponding
        class should remain unchanged.
        :return:
        """
        self.desired_dfs = pd.DataFrame()  # empty this before appending new dataframes
        expected_populations = {}

        if desired_ratios:
            validate_input(class_population=self.class_population, desired_ratios=desired_ratios)
            total_population = np.sum(self.class_population.values())

            for label, value in desired_ratios.items():
                if value == -1:
                    expected_populations[label] = self.class_population[label]
                else:
                    expected_populations[label] = np.round(value * total_population)

        else:
            validate_input(class_population=self.class_population,
                           desired_populations=desired_populations)
            for label, value in desired_populations.items():
                if value == -1:
                    expected_populations[label] = self.class_population[label]
                else:
                    expected_populations[label] = desired_populations[label]

        for label, value in expected_populations.items():
            sample_size = np.round(expected_populations[label])
            output_dfs = self.sample_each_class(self.dfs_dict[label], sample_size)
            self.desired_dfs = self.desired_dfs.append(output_dfs)

        return self.desired_dfs

    def undersample(self, minority_labels: list, majority_labels: list, base_minority: str):
        """
        Undersamples form the majority class to achieve a 1:1 balance between the minority and
        majority classes. This is done by keeping the population of the base_minority unchanged,
        make all other minority classes to have an equal population, and then reduce the population
        of the majority classes to match with the minority classes. This reduction is done in
        a way that all majority classes reach to an equal population, and their total population
        sum up to the total population of the minority classes. Hence a 1:1 balance.

        Example: Consider mvts data with 5 classes:

            |A| = 100, |B| = 400, |C| = 300, |D| = 700, |E| = 2000
            |A| + |B| = 500, |C| + |D| + |E| = 3000

        and given is: minority_labels = ['A', 'B'], majority_labels = ['C', 'D', 'E'], base_minority = 'A'

        then, the resultant dataframe would have the following populations:

            |A| = 100, |B| = 100, |C| = 200/3, |D| = 200/3, |E| = 200/3
            |A| + |B| = 200, |C| + |D| + |E| = 200

        :param minority_labels:
        :param majority_labels:
        :param base_minority:
        :return:
        """
        validate_sampling_input(self.class_population, minority_labels=minority_labels,
                                majority_labels=majority_labels
                                , base=base_minority)
        if base_minority:
            base_count = self.dfs_dict[base_minority].shape[0]
            total_min_count = base_count * len(minority_labels)
            for label in minority_labels:
                if label != base_minority:
                    output_dfs = self.sample_each_class(self.dfs_dict[label], base_count)
                    self.desired_dfs = self.desired_dfs.append(output_dfs)

                else:
                    self.desired_dfs = self.desired_dfs.append(self.dfs_dict[label])

            maj_base_count = round(total_min_count / len(majority_labels))
            for label in majority_labels:
                output_dfs = self.sample_each_class(self.dfs_dict[label], maj_base_count)
                self.desired_dfs = self.desired_dfs.append(output_dfs)

            # TODO: return?

    def oversample(self, minority_labels: list, majority_labels: list, base_majority: str):
        """
        Oversamples form the minority class to achieve a 1:1 balance between the minority and
        majority classes. This is done by keeping the population of the base_majority unchanged,
        make all other majority classes to have an equal population, and then increase the population
        of the minority classes to match with the majority classes. This increase is done in
        a way that all minority classes reach to an equal population, and their total population
        sums up to the total population of the majority classes. Hence a 1:1 balance.

        Example: Consider mvts data with 5 classes:

            |A| = 100, |B| = 400, |C| = 300, |D| = 700, |E| = 2000
            |A| + |B| = 500, |C| + |D| + |E| = 3000

        and given is: minority_labels = ['A', 'B'], majority_labels = ['C', 'D', 'E'], base_majority = 'D'

        then, the resultant dataframe would have the following populations:

            |A| = 2100/2, |B| = 2100/2, |C| = 700, |D| = 700, |E| = 700
            |A| + |B| = 2100, |C| + |D| + |E| = 2100

        :param minority_labels:
        :param majority_labels:
        :param base_majority:
        :return:
        """
        validate_sampling_input(self.class_population, minority_labels=minority_labels,
                                majority_labels=majority_labels,
                                base=base_majority)
        if base_majority:
            base_count = self.dfs_dict[base_majority].shape[0]
            total_maj_count = base_count * len(majority_labels)
            for label in majority_labels:
                if label != base_majority:
                    output_dfs = self.sample_each_class(self.dfs_dict[label], base_count)
                    self.desired_dfs = self.desired_dfs.append(output_dfs)

                else:
                    self.desired_dfs = self.desired_dfs.append(self.dfs_dict[label])

            min_base_count = round(total_maj_count / len(minority_labels))
            for label in minority_labels:
                output_dfs = self.sample_each_class(self.dfs_dict[label], min_base_count)
                self.desired_dfs = self.desired_dfs.append(output_dfs)

        # TODO: return?

    def sample_each_class(self, input_dfs: pd.DataFrame, new_sample_size: int) -> pd.DataFrame:
        """
        This method samples `new_sample_size` instances from a given dataframe. If the desired
        sample size is larger than the original population (i.e., `new_sample_size >
        input_dfs.shape[0]`), then the entire population will be used, as well as the extra samples
        needed to achieve the desired sample size. The extra instances will be sampled with
        replacement.
        :param input_dfs: The input dataset to sample from.
        :param new_sample_size: The size of the desired sample.
        :return: The sampled dataframe.
        """
        population_size = input_dfs.shape[0]
        if new_sample_size > population_size:
            extra_samples = input_dfs.sample(new_sample_size - population_size, replace=False)
            output_dfs = input_dfs.append(extra_samples, ignore_index=True)
        else:
            output_dfs = input_dfs.sample(new_sample_size, replace=False)
        return output_dfs


def main():
    import os
    import pandas as pd
    import CONSTANTS as CONST

    path_to_extracted_features = os.path.join(CONST.ROOT,
                                              'tests/test_dataset/extracted_features/extracted_features_TEST_INPUT.csv')

    df = pd.read_csv(path_to_extracted_features, sep='\t')
    sampler = Sampler(df, 'lab')

    print('desired dfs', sampler.desired_dfs)
    print('class labels', sampler.labels)
    print('class population:\n', sampler.class_population)
    print('class ratios:\n', sampler.class_ratios)
    # print('class decomposed df:\n', sampler.dfs_dict)

    print(sampler.desired_dfs.shape)
    desired_populations = {'NF': -1, 'C': 10}
    sampler.sample(desired_populations=desired_populations)
    print(sampler.desired_dfs.shape)


if __name__ == "__main__":
    main()
