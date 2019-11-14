import pandas as pd
from sampling.input_validator import validate_input,validate_sampling_input


class Sampler:
    """
    This module  contains several sampling methods to handle dataset issues like imbalanced class.

    See ReadMe.md for usage example.
    """

    def __init__(self, mvts_df: pd.DataFrame, label_col_name):
        """
        The constructor method takes the input dataset as pandas dataframe and the column-name to
        decompose the specific column or feature. We can get the population and ratio of the
        provided column-names. It also decomposes the whole dataset in several smaller
        dataset based on class.

        :param mvts_df: a dataframe whose rows are the instances and columns are the features.
        :param label_col_name: the name of the column that is used to keep the class labels.
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
        class_ratios = [self.class_population[label] / total for label in self.class_population.keys()]
        return class_ratios

    def __decompose_mvts(self):
        """
        A private method that decomposes the mvts dataframe into several dataframes, one for each
        class label.
        """
        for key in self.dfs_dict.keys():
            self.dfs_dict[key] = self.df[:][self.df[self.label_col_name] == key]

    def get_labels(self) -> pd.DataFrame:
        """
        :return: a list of all class labels of the data.
        """
        return self.labels

    def get_original_populations(self) -> dict:
        """
        Gets the per-class population of the original dataset.

        :return: a dictionary of labels (as key) and class populations (as values)
        """
        return self.class_population

    def get_original_ratios(self) -> list:
        """
        Gets the per-class ratio of the original dataset.

        :return: a dictionary of labels (as key) and class ratios (as values)
        """
        return self.class_ratios

    def get_decomposed_mvts(self) -> dict:
        """
        Decomposes the dataframe provided to the class into several dataframes, one for each
        class label.

        :return: a dictionary of dataframes, with keys being the class labels, and the corresponding
                 dataframes contain instances of those labels only.
        """
        return self.dfs_dict

    def sample(self, desired_populations: dict = None, desired_ratios: dict = None) -> pd.DataFrame:
        """
        Using this method one could do either undersampling or oversampling, in the most generic
        fashion. That is, the user determines the expected population size or ratios that they
        would like to get from the mvts data.

        Example: Consider a mvts data with 5 classes::

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
        :return: the sampled data.
        """

        if not desired_populations:
            validate_input(self.class_population, desired_ratios=desired_ratios)
            total_population = sum(self.class_population.values())
            desired_populations = {}

            for label, value in desired_ratios.items():
                if value == -1:
                    desired_populations[label] = self.class_population[label]
                else:
                    desired_populations[label] = round(value * total_population)

        else:

            validate_input(self.class_population, desired_populations=desired_populations)

        for label in self.labels:
            if desired_populations[label] == -1:
                self.desired_dfs = self.desired_dfs.append(self.dfs_dict[label])
            elif value >= 0:
                count = round(desired_populations[label])
                output_dfs = self.sample_each_class(self.dfs_dict[label], count)
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

        Example: Consider mvts data with 5 classes::

            |A| = 100, |B| = 400, |C| = 300, |D| = 700, |E| = 2000
            |A| + |B| = 500, |C| + |D| + |E| = 3000

        and given is: minority_labels = ['A', 'B'], majority_labels = ['C', 'D', 'E'],
        base_minority = 'A', then, the resultant dataframe would have the following populations::

            |A| = 100, |B| = 100, |C| = 200/3, |D| = 200/3, |E| = 200/3
            |A| + |B| = 200, |C| + |D| + |E| = 200

        :param minority_labels:
        :param majority_labels:
        :param base_minority:
        :return:
        """
        validate_sampling_input(self.class_population,minority_labels= minority_labels, majority_labels= majority_labels
                                , base= base_minority)
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

    def oversample(self, minority_labels: list, majority_labels: list, base_majority: str):
        """
        Oversamples form the minority class to achieve a 1:1 balance between the minority and
        majority classes. This is done by keeping the population of the base_majority unchanged,
        make all other majority classes to have an equal population, and then increase the
        population of the minority classes to match with the majority classes. This increase is
        done in a way that all minority classes reach to an equal population, and their total
        population sums up to the total population of the majority classes. Hence a 1:1
        balance.

        Example: Consider mvts data with 5 classes::

            |A| = 100, |B| = 400, |C| = 300, |D| = 700, |E| = 2000
            |A| + |B| = 500, |C| + |D| + |E| = 3000

        and given is: minority_labels = ['A', 'B'], majority_labels = ['C', 'D', 'E'],
        base_majority = 'D', then, the resultant dataframe would have the following populations::

            |A| = 2100/2, |B| = 2100/2, |C| = 700, |D| = 700, |E| = 700
            |A| + |B| = 2100, |C| + |D| + |E| = 2100

        :param minority_labels:
        :param majority_labels:
        :param base_majority:
        :return:
        """
        validate_sampling_input(self.class_population, minority_labels=minority_labels, majority_labels=majority_labels,
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

    def sample_each_class(self, input_dfs: pd.DataFrame, count: int) -> pd.DataFrame:
        """
        This method returns output samples based on given dataset and count of desired output
        sample. If desired sample size is more than input dataset then whole input dataset is
        used and rest of the samples are created using sampling with replacement from the input
        dataset.

        :param input_dfs: Input Dataset
        :param count: Count of desired output samples
        :return: Sampled Dataset
        """

        if count > input_dfs.shape[0]:
            output_dfs = input_dfs.append(
                input_dfs.sample((count - input_dfs.shape[0]), replace=True))
        else:
            output_dfs = input_dfs.sample(count)

        return output_dfs
