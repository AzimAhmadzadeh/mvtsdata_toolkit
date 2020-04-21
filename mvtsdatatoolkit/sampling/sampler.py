import pandas as pd
import numpy as np
from mvtsdatatoolkit.sampling.input_validator import validate_sampling_input, \
    validate_under_over_sampling_input


def _extract_labels(mvts: pd.DataFrame, label_col_name) -> list:
    """
    A private method that extracts the unique class labels from the dataframe.

    :return: A list of unique class labels.
    """
    class_labels = mvts[label_col_name].unique().tolist()
    return class_labels


def _decompose_mvts(mvts: pd.DataFrame, class_labels: list, label_col_name) -> dict:
    """
    A private method that decomposes the MVTS dataframe into several dataframes, one for each
    class label.

    :return: A dictionary of labels (as keys) and dataframes (as values), each corresponding
             to one label.
     """
    decomposed_mvts: dict = {label: pd.DataFrame for label in class_labels}
    for key in decomposed_mvts.keys():
        decomposed_mvts[key] = mvts[:][mvts[label_col_name] == key]
    return decomposed_mvts


def _compute_populations(mvts: pd.DataFrame, label_col_name) -> dict:
    """
    A private method that computes the population corresponding to each class label.

    :param mvts: The dataframe who class population is of interest.
    :param label_col_name: The column-name corresponding to the class labels in `mvts`.

    :return: A dictionary of class labels (as keys) and class populations (as values).
    """
    class_labels: list = _extract_labels(mvts, label_col_name)
    decomposed_mvts = _decompose_mvts(mvts, class_labels, label_col_name)
    pop_dict = {label: len(decomposed_mvts[label]) for label in decomposed_mvts.keys()}
    return pop_dict


def _compute_ratios(mvts: pd.DataFrame, label_col_name) -> dict:
    """
    A private method that computes the population corresponding to each class label.

    :param mvts: The dataframe who class population is of interest.
    :param label_col_name: The column-name corresponding to the class labels in `mvts`.

    :return: A dictionary of class labels (as keys) and class populations (as values).
    """

    class_labels: list = _extract_labels(mvts, label_col_name)
    decomposed_mvts = _decompose_mvts(mvts, class_labels, label_col_name)
    pop_dict = {label: len(decomposed_mvts[label]) for label in decomposed_mvts.keys()}
    return pop_dict


class Sampler:
    """
    This module contains several methods that assist sampling for a number of purposes,
    among which, to remedy the class-imbalance issue, is the primary objective.
    """

    def __init__(self, extracted_features_df: pd.DataFrame, label_col_name):
        """
        The constructor method takes the input dataset and the column name corresponding to the
        class labels, and extracts the information about class labels and class populations.

        :param extracted_features_df: An MVTS dataframe that needs normalization.
        :param label_col_name: Column-name corresponding to the class class_labels.
        """
        self.label_col_name = label_col_name
        self.original_mvts = extracted_features_df
        self.sampled_mvts = pd.DataFrame()
        self.class_labels: list = _extract_labels(self.original_mvts, self.label_col_name)
        self.__decomposed_original_mvts: dict = _decompose_mvts(self.original_mvts,
                                                                self.class_labels,
                                                                self.label_col_name)
        self.__decomposed_sampled_mvts: dict = {}

        self.original_class_populations: dict = {}
        self.original_class_ratios: dict = {}
        self.sampled_class_populations: dict = {}
        self.sampled_class_ratios: dict = {}

        self.__update_original_metadata()

    def __compute_original_population(self) -> dict:
        """
        A private method that computes the population of the original dataset.

        :return: A dictionary of class labels (as keys) and class populations (as values).
        """
        pop_dict = {label: len(self.__decomposed_original_mvts[label]) for label in
                    self.__decomposed_original_mvts.keys()}
        return pop_dict

    def __compute_original_ratios(self) -> dict:
        """
        A private method that computes the class ratios of the original dataset.
        """
        total = sum(self.original_class_populations.values())
        class_ratios = {label: self.original_class_populations[label] / total for label in
                        self.class_labels}
        return class_ratios

    def __compute_sampled_population(self) -> dict:
        """
        A private method that computes the class population of the sampled dataset.

        Note: Do not use this method directly. Instead, call `update_after_sampling`.

        :return: A dictionary of class labels (as keys) and class populations (as values).
        """
        if self.sampled_mvts.empty:
            return {}

        self.__decomposed_sampled_mvts = _decompose_mvts(self.sampled_mvts, self.class_labels,
                                                         self.label_col_name)
        pop_dict = {label: len(self.__decomposed_sampled_mvts[label]) for label in
                    self.__decomposed_sampled_mvts.keys()}
        return pop_dict

    def __compute_sampled_ratios(self) -> dict:
        """
        A private method that computes the class ratios of the sampled dataset.

        Note: Do not use this method directly. Instead, call `update_after_sampling`.

        :return: A dictionary of class labels (as keys) and class ratios (as values).
        """
        total = sum(self.sampled_class_populations.values())
        class_ratios = {label: self.sampled_class_populations[label] / total
                        for label in self.class_labels}
        return class_ratios

    def get_labels(self) -> list:
        """
        A getter method for the class_labels.

        :return: The class field `class_labels`; a list of all class labels in the data.
        """
        return self.class_labels

    def __update_original_metadata(self):
        self.original_class_populations = self.__compute_original_population()
        self.original_class_ratios = self.__compute_original_ratios()

    def __update_sampled_metadata(self):
        self.sampled_class_populations = self.__compute_sampled_population()
        self.sampled_class_ratios = self.__compute_sampled_ratios()

    def sample(self, desired_populations: dict = None, desired_ratios: dict = None):
        """
        Using this method one could do either undersampling or oversampling, in the most generic
        fashion. That is, the user determines the expected population size or ratios that they
        would like to get from the MVTS data.
        Example: Consider an MVTS data with 5 classes:

            |A| = 100, |B| = 400, |C| = 300, |D| = 700, |E| = 2000

        and given is: desired_ratios = [-1, -1, 0.33, 0.33, 0.33]

        Then, the instances of classes A and B will not change, while
        the population size of each of the  C, D, and E classes would be
        one-third of the sample size (3500 * 0.33).

        Note:
            1. One and only one of the arguments must be provided.
            2. The dictionary must contain all class class_labels present in the mvts dataframe.
            3. The number -1 can be used wherever the population or ratio should not change.

        :param desired_populations: A dictionary of label-integer pairs, where each integer specifies
        the desired population of the corresponding class. The integers must be positive, but -1
        can be used to indicate that the population of the corresponding class should remain
        unchanged.
        :param desired_ratios: A dictionary of label-float pairs, where each float specifies
        the desired ratios (with respect to the total sample size) of the corresponding class.
        The floats must be positive, but -1 can be used to indicate that the ratio of the
        corresponding class should remain unchanged.
        """
        self.sampled_mvts = pd.DataFrame()  # empty this before appending new dataframes
        expected_populations = {}

        if desired_ratios:
            validate_sampling_input(class_populations=self.original_class_populations,
                                    desired_ratios=desired_ratios)
            total_population = sum(self.original_class_populations.values())

            for label, value in desired_ratios.items():
                if value == -1:
                    expected_populations[label] = self.original_class_populations[label]
                else:
                    expected_populations[label] = np.round(value * total_population)

        else:
            validate_sampling_input(class_populations=self.original_class_populations,
                                    desired_populations=desired_populations)
            for label, value in desired_populations.items():
                if value == -1:
                    expected_populations[label] = self.original_class_populations[label]
                else:
                    expected_populations[label] = desired_populations[label]

        for label, value in expected_populations.items():
            sample_size = int(expected_populations[label])
            output_dfs = self.sample_each_class(self.__decomposed_original_mvts[label], sample_size)
            self.sampled_mvts = self.sampled_mvts.append(output_dfs)

        self.__update_sampled_metadata()
        return self.sampled_mvts

    def undersample(self, minority_labels: list, majority_labels: list, base_minority: str):
        """
        Undersamples from the majority classes to achieve a 1:1 balance between the minority and
        majority classes. This is done in such a way that the outcome follows these criteria:

         * The minority classes have an equal population, equal to that of `base_minority` class.
         * The majority classes have an equal population, such that the next criterion is held true.
         * Total population of the majority classes is (undersampled to become) equal to the total
         population of the minority classes.

        Example: Consider an mvts dataset with 5 classes, A, B, C, D, and E:

            |A| = 100, |B| = 400, |C| = 300, |D| = 700, |E| = 2000

            where

            |A| + |B| = 500, |C| + |D| + |E| = 3000

        and suppose given is::

            minority_labels = ['A', 'B'], majority_labels = ['C', 'D', 'E'], base_minority = 'A'


        Then, the sampled dataframe would have the following populations:

            |A| = 100, |B| = 100, |C| = 200/3, |D| = 200/3, |E| = 200/3

            where

            |A| + |B| = 200, |C| + |D| + |E| = 200

        :param minority_labels: A list of class labels considered to be the minority classes.
        :param majority_labels: A list of class labels considered to be the majority classes.
        :param base_minority: The class label based on which, the sampling method is decided.
        """
        validate_under_over_sampling_input(self.original_class_populations,
                                           minority_labels=minority_labels,
                                           majority_labels=majority_labels,
                                           base_minority=base_minority)
        if base_minority:
            base_count = self.__decomposed_original_mvts[base_minority].shape[0]
            total_min_count = base_count * len(minority_labels)
            for label in minority_labels:
                if label != base_minority:
                    output_dfs = self.sample_each_class(self.__decomposed_original_mvts[label],
                                                        base_count)
                    self.sampled_mvts = self.sampled_mvts.append(output_dfs)

                else:
                    self.sampled_mvts = self.sampled_mvts.append(
                        self.__decomposed_original_mvts[label])

            maj_base_count = round(total_min_count / len(majority_labels))
            for label in majority_labels:
                output_dfs = self.sample_each_class(self.__decomposed_original_mvts[label],
                                                    maj_base_count)
                self.sampled_mvts = self.sampled_mvts.append(output_dfs)

        self.__update_sampled_metadata()
        return self.sampled_mvts

    def oversample(self, minority_labels: list, majority_labels: list, base_majority: str):
        """
        Oversamples from the majority classes to achieve a 1:1 balance between the minority and
        majority classes. This is done in such a way that the outcome follows these criteria:

         * The minority classes have an equal population, equal to that of `base_minority` class.
         * The majority classes have an equal population, such that the next criterion is held true.
         * Total population of the majority classes is (oversampled to become) equal to the total
         population of the minority classes.


        Example: Consider mvts data with 5 classes:

            |A| = 100, |B| = 400, |C| = 300, |D| = 700, |E| = 2000

            where

            |A| + |B| = 500, |C| + |D| + |E| = 3000

        and given is::

            minority_labels = ['A', 'B'], majority_labels = ['C', 'D', 'E'], base_majority = 'D'.

        Then, the sampled dataframe would have the following populations:

            |A| = 2100/2, |B| = 2100/2, |C| = 700, |D| = 700, |E| = 700

            where

            |A| + |B| = 2100, |C| + |D| + |E| = 2100.

        """
        validate_under_over_sampling_input(self.original_class_populations,
                                           minority_labels=minority_labels,
                                           majority_labels=majority_labels,
                                           base_majority=base_majority)
        if base_majority:
            base_count = self.__decomposed_original_mvts[base_majority].shape[0]
            total_maj_count = base_count * len(majority_labels)
            for label in majority_labels:
                if label != base_majority:
                    output_dfs = self.sample_each_class(self.__decomposed_original_mvts[label],
                                                        base_count)
                    self.sampled_mvts = self.sampled_mvts.append(output_dfs)

                else:
                    self.sampled_mvts = self.sampled_mvts.append(
                        self.__decomposed_original_mvts[label])

            min_base_count = round(total_maj_count / len(minority_labels))
            for label in minority_labels:
                output_dfs = self.sample_each_class(self.__decomposed_original_mvts[label],
                                                    min_base_count)
                self.sampled_mvts = self.sampled_mvts.append(output_dfs)

        self.__update_sampled_metadata()
        return self.sampled_mvts

    def sample_each_class(self, input_dfs: pd.DataFrame, new_sample_size: int) -> pd.DataFrame:
        """
        This method samples `new_sample_size` instances from a given dataframe. If the desired
        sample size is larger than the original population (i.e., `new_sample_size >
        input_dfs.shape[0]`), then the entire population will be used, as well as the extra samples
        needed to achieve the desired sample size. The extra instances will be sampled from
        `input_dfs` with replacement.

        :param input_dfs: The input dataset to sample from.
        :param new_sample_size: The size of the desired sample.

        :return: The sampled dataframe.
        """
        new_sample_size = int(new_sample_size)
        population_size = input_dfs.shape[0]
        if new_sample_size > population_size:
            extra_samples = input_dfs.sample(new_sample_size - population_size, replace=True)
            output_dfs = input_dfs.append(extra_samples, ignore_index=True)
        else:
            output_dfs = input_dfs.sample(new_sample_size, replace=False)
        return output_dfs
