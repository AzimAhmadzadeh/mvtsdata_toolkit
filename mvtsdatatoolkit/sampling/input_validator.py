import numpy as np


def validate_sampling_input(class_populations: dict, desired_ratios: dict = None,
                            desired_populations: dict = None):
    """
    This method validates the three arguments against the following rules:

     - Both `desired_ratios` and `desired_populations` cannot be `None` at the same time.
     - Both `desired_ratios` and `desired_populations` cannot be given at the same time.
     - Class labels in `desired_populations` must match with those in `class_populations`.
     - Class labels in `desired_ratios` must match with those in `class_populations`.
     - Class populations in `desired_populations` MUST be either positive or -1.
     - Class ratios in `desired_ratios` MUST be either positive or -1. The positive values may be
       larger than 1.0.

    :param class_populations: The class labels present in the data.
    :param desired_ratios: The desired ratios of each class to be sampled.
    :param desired_populations: The desired population of each class to be sampled.

    :return: True, if no exception was raised.
    """
    desired_ratios = desired_ratios if desired_ratios else {}
    desired_populations = desired_populations if desired_populations else {}

    if not desired_populations and not desired_ratios:
        # at least one MUST be given
        raise ValueError(
            """
            One and only one of the of the args, `desired_populations` or `desired_ratios` MUST 
            be given! None is given!
            """
        )

    if desired_populations and desired_ratios:
        # both CANNOT be given at the same time
        raise ValueError(
            """
            One and only one of the of the args, `desired_populations` or `desired_ratios` MUST 
            be given! Both are given!
            """
        )

    if desired_ratios:
        # class labels MUST match.
        if set(class_populations.keys()) != set(desired_ratios.keys()):
            raise ValueError(
                """
                The class labels in `desired_ratios` do not match with those specified in 
                `class-populations`!
                """
            )
        n_of_invalid_ratios = \
            np.sum([True for _, count in desired_ratios.items() if count != -1 and count < 0])
        # no negative count (except -1) is allowed.
        if n_of_invalid_ratios > 0:
            raise ValueError(
                """
                The dictionary `desired_ratios` CANNOT have negative values except -1!
                """
            )

    if desired_populations:
        # class labels MUST match.
        if set(class_populations.keys()) != set(desired_populations.keys()):
            raise ValueError(
                """
                The class labels in `desired_populations` do not match with those specified in 
                `class-populations`!
                """
            )

        n_of_nonint_populations = \
            len([True for _, count in desired_populations.items() if count != int(count)])
        if n_of_nonint_populations > 0:
            # no decimal is allowed.
            raise ValueError(
                """
                The dictionary `desired_populations` CANNOT have non-int values!
                """
            )

        n_of_invalid_populations = \
            len([True for _, count in desired_populations.items() if count < -1])
        if n_of_invalid_populations > 0:
            # no negative count (except -1) is allowed.
            raise ValueError(
                """
                The dictionary `desired_populations` CANNOT have negative values except -1!
                """
            )

    return True


def validate_under_over_sampling_input(class_populations, minority_labels, majority_labels,
                                       base_minority=None, base_majority=None):
    """
    This is to validate the arguments of two methods in the class `Sampler` in
    `sampling.sampler`, namely `undersample` and `oversample`.
    :param class_populations: See the corresponding docstring in `Sampler`.
    :param minority_labels: See the corresponding docstring in `Sampler`.
    :param majority_labels: See the corresponding docstring in `Sampler`.
    :param base_majority: See the corresponding docstring in `Sampler`.
    :param base_minority: See the corresponding docstring in `Sampler`.

    :return: True, if no exception was raised.
    """
    union_of_labels = set(minority_labels).union(set(majority_labels))
    if union_of_labels != set(class_populations):
        # union of minority and majority classes must contain all classes.
        raise ValueError(
            """
            One or more class labels are not present in either of the dictionaries, 
            `minority_labels` or `majority_labels`!
            """
        )
    intersection_of_labels = set(minority_labels).intersection(set(majority_labels))
    if len(intersection_of_labels) > 0:
        # no intersection of labels allowed.
        raise ValueError(
            """
            The dictionaries, `minority_labels` and `majority_labels`, MUST be mutually exclusive!
            """
        )

    if base_majority:
        if base_majority not in set(majority_labels):
            # base_minority should be a minority
            raise ValueError(
                """
                The (majority) base label MUST be one of the labels in `majority_labels`! '{}' is
                not!
                """.format(base_majority)
            )
    if base_minority:
        if base_minority not in set(minority_labels):
            # base majority must be a majority
            raise ValueError(
                """
                he (minority) base lanel MUST be one of the labels in `minority_labels`! '{}' is 
                not!
                """.format(base_minority)
            )

    return True
