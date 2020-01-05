

def validate_input(class_population: dict = None, desired_ratios: dict = None,
                   desired_populations: dict = None):
    """

    :param class_population: Existing classes with population present in dataset
    :param desired_ratios: Given input by user(class label with desired ratio)
    :param desired_populations: Given input by user(class label with desired population)
    :return:
    """
    if not desired_populations and not desired_ratios:
        raise ValueError(
            """
            At least one of the args, `desired_population` or `desired_ratios` MUST be given!
            """
        )
    elif not desired_populations:

        # Check whether the argument passed by the user matches with the dataset
        if set(class_population.keys()) != set(desired_ratios.keys()):
            raise ValueError(
                """
                The keys of the passed dictionaries, `class_population` and `desired_ratios` do 
                not match!
                """
            )
    elif not desired_ratios:
        # Check whether the argument passed by the user matches with the dataset
        if set(class_population.keys()) != set(desired_populations.keys()):
            raise ValueError(
                """
                The keys of the passed dictionaries, `class_population` and `desired_populations` do 
                not match!
                """
            )
        if (desired_populations.values() != -1) and (desired_populations.values() <= 0) and (
                desired_populations.values() > 1):
            raise ValueError(
                """
                Please enter values between 0 to 1 or -1 in desired ratios
                """
            )

    else:
        raise ValueError(
            """
            One argument needs to be passed
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
