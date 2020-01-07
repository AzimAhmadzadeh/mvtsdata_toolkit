import numpy as np


def validate_input(class_population: dict, desired_ratios: dict = None,
                   desired_populations: dict = None):
    """

    :param class_population: Existing classes with population present in dataset
    :param desired_ratios: Given input by user(class label with desired ratio)
    :param desired_populations: Given input by user(class label with desired population)
    :return:
    """
    desired_ratios = desired_ratios if desired_ratios else {}
    desired_populations = desired_populations if desired_populations else {}

    if not desired_populations and not desired_ratios:
        # at least one MUST be given
        raise ValueError(
            """
            At least one of the args, `desired_population` or `desired_ratios` MUST be given!
            """
        )
    if desired_ratios:
        # class labels MUST match.
        if set(class_population.keys()) != set(desired_ratios.keys()):
            raise ValueError(
                """
                The keys of the passed dictionaries, `class_population` and `desired_ratios` do 
                not match!
                """
            )
        invalid_count = \
            np.sum([True for _, count in desired_ratios.items() if count != -1 and count < 0])
        # no negative count (except -1) is allowed.
        if invalid_count:
            raise ValueError(
                """
                The dictionary `desired_ratios` CANNOT have negative values except -1.
                """
            )
    if desired_populations:
        # class labels MUST match.
        if set(class_population.keys()) != set(desired_populations.keys()):
            raise ValueError(
                """
                The class labels in `desired_populations` do not match with those specified in 
                `class-populations`!
                """
            )
        invalid_count = \
            np.sum([True for _, count in desired_populations.items() if count != -1 and count < 0])
        # no negative count (except -1) is allowed.
        if invalid_count:
            raise ValueError(
                """
                The dictionary `desired_populations` CANNOT have negative values except -1.
                """
            )


def validate_sampling_input(class_population, minority_labels, majority_labels, base):
    """
    :param class_population: Existing classes with population present in dataset
    :param minority_labels: Given input by user(class labels of minority set)
    :param majority_labels: Given input by user(class labels of majority set)
    :param base: Given input by user(Base minority r majority class label)
    :return:
    """
    all_classes = set(minority_labels).union(set(majority_labels))
    if all_classes != set(class_population):
        raise ValueError(
            """
            Please enter correct labels and desired population
            """
        )
    if base not in all_classes:
        raise ValueError(
            """
            Please enter correct base value.
            """
        )
    if set(minority_labels).intersection(set(majority_labels)):
        raise ValueError(
            """
            Majority and Minority Classes should have distinct class labels.
            """
        )
