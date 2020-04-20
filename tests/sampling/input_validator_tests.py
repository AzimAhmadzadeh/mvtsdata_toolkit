import unittest
from mvtsdatatoolkit.sampling import input_validator


class TestInputValidator(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        pass

    @classmethod
    def tearDownClass(cls) -> None:
        pass

    def test_sampling_input_x1(self):
        """ Tests if the method raises a proper exception if none of the two optional arguments
        are given."""
        population = {'A': 10, 'B': 90}
        desired_ratios = None                        # <---- both None
        desired_populations = None                   # <---- both None
        with self.assertRaises(ValueError):
            input_validator.validate_sampling_input(class_populations=population,
                                                    desired_ratios=desired_ratios,
                                                    desired_populations=desired_populations)

    def test_sampling_input_x2(self):
        """ Tests if the method raises a proper exception if the class labels do not match in
        `class_populations` and `desired_populations`."""
        population = {'A': 10, 'B': 90}
        desired_ratios = {'A': 0.5, 'C': 0.2}                   # <---- invalid label C
        desired_populations = None
        with self.assertRaises(ValueError):
            input_validator.validate_sampling_input(class_populations=population,
                                                    desired_ratios=desired_ratios,
                                                    desired_populations=desired_populations)

    def test_sampling_input_x3(self):
        """ Tests if the method raises a proper exception if the class labels do not match in
        `class_populations` and `desired_ratios`."""
        population = {'A': 10, 'B': 90}
        desired_ratios = None
        desired_populations = {'A': 5, 'C': 20}                   # <---- invalid label C
        with self.assertRaises(ValueError):
            input_validator.validate_sampling_input(class_populations=population,
                                                    desired_ratios=desired_ratios,
                                                    desired_populations=desired_populations)

    def test_sampling_input_x4(self):
        """ Tests if the method raises a proper exception if both of the optional arguments,
        `desired_ratios` and `desired_populations`, are given."""
        population = {'A': 10, 'B': 90}
        desired_ratios = {'A': 0.5, 'B': 0.2}                   # <---- both given
        desired_populations = {'A': 5, 'B': 20}                 # <---- both given
        with self.assertRaises(ValueError):
            input_validator.validate_sampling_input(class_populations=population,
                                                    desired_ratios=desired_ratios,
                                                    desired_populations=desired_populations)

    def test_sampling_input_x5(self):
        """ Tests if the method raises a proper exception if `desired_populations` contains
        non-int values (float is allowed as long as its decimal is zero)."""
        population = {'A': 10, 'B': 90}
        desired_ratios = None
        desired_populations = {'A': 5, 'B': 20.1}                   # <---- non-int B
        with self.assertRaises(ValueError):
            input_validator.validate_sampling_input(class_populations=population,
                                                    desired_ratios=desired_ratios,
                                                    desired_populations=desired_populations)

    def test_sampling_input_x6(self):
        """ Tests if the method raises a proper exception if `desired_populations` contains
        negative values (except -1)."""
        population = {'A': 10, 'B': 90}
        desired_ratios = None
        desired_populations = {'A': -5, 'B': 20}                   # <---- negative A
        with self.assertRaises(ValueError):
            input_validator.validate_sampling_input(class_populations=population,
                                                    desired_ratios=desired_ratios,
                                                    desired_populations=desired_populations)

    def test_sampling_input_x7(self):
        """ Tests if the method accepts -1 in its `desired_populations`."""
        population = {'A': 10, 'B': 90}
        desired_ratios = None
        desired_populations = {'A': -1, 'B': 20}                    # <---- -1 OK
        expected = True
        actual = input_validator.validate_sampling_input(class_populations=population,
                                                         desired_ratios=desired_ratios,
                                                         desired_populations=desired_populations)
        self.assertEqual(actual, expected)

    def test_sampling_input_x8(self):
        """ Tests if the method raises a proper exception if `desired_ratios` contains
        negative values (except -1)."""
        population = {'A': 10, 'B': 90}
        desired_ratios = {'A': -0.5, 'B': 0.1}                   # <---- negative A
        desired_populations = None
        with self.assertRaises(ValueError):
            input_validator.validate_sampling_input(class_populations=population,
                                                    desired_ratios=desired_ratios,
                                                    desired_populations=desired_populations)

    def test_sampling_input_x9(self):
        """ Tests if the method accepts -1 in its `desired_ratios`."""
        population = {'A': 10, 'B': 90}
        desired_ratios = {'A': -1.0, 'B': 0.2}                   # <---- -1 OK
        desired_populations = None
        expected = True
        actual = input_validator.validate_sampling_input(class_populations=population,
                                                         desired_ratios=desired_ratios,
                                                         desired_populations=desired_populations)
        self.assertEqual(expected, actual)

    def test_under_over_sampling_input_x1(self):
        """ Tests if the method raises proper exception when minority and majority labels
        intersect."""
        population = {'A': 10, 'B': 5, 'X': 50, 'Y': 70}
        minority_labels = ['A', 'B']                                # <---- intersect at B
        majority_labels = ['X', 'Y', 'B']                           # <---- intersect at B
        base_minority = 'A'
        base_majority = 'Y'
        with self.assertRaises(ValueError):
            input_validator.validate_under_over_sampling_input(class_populations=population,
                                                               minority_labels=minority_labels,
                                                               majority_labels=majority_labels,
                                                               base_minority=base_minority,
                                                               base_majority=base_majority)

    def test_under_over_sampling_input_x2(self):
        """ Tests if the method raises proper exception when the union of minority and majority 
        labels does not match with the class labels."""
        population = {'A': 10, 'B': 5, 'X': 50, 'Y': 70}
        minority_labels = ['A', 'B']
        majority_labels = ['Y']                                 # <---- missing X
        base_minority = 'A'
        base_majority = 'Y'
        with self.assertRaises(ValueError):
            input_validator.validate_under_over_sampling_input(class_populations=population,
                                                               minority_labels=minority_labels,
                                                               majority_labels=majority_labels,
                                                               base_minority=base_minority,
                                                               base_majority=base_majority)

    def test_under_over_sampling_input_x3(self):
        """ Tests if the method raises proper exception when the base_minority does not
        belong to the minority_labels."""
        population = {'A': 10, 'B': 5, 'X': 50, 'Y': 70}
        minority_labels = ['A', 'B']
        majority_labels = ['X', 'Y']
        base_minority = 'Y'                                 # <---- Should be A or B
        base_majority = 'X'
        with self.assertRaises(ValueError):
            input_validator.validate_under_over_sampling_input(class_populations=population,
                                                               minority_labels=minority_labels,
                                                               majority_labels=majority_labels,
                                                               base_minority=base_minority,
                                                               base_majority=base_majority)

    def test_under_over_sampling_input_x4(self):
        """ Tests if the method raises proper exception when the base_majority does not
        belong to the majority_labels."""
        population = {'A': 10, 'B': 5, 'X': 50, 'Y': 70}
        minority_labels = ['A', 'B']
        majority_labels = ['X', 'Y']
        base_minority = 'A'
        base_majority = 'B'                                 # <---- Should be X or Y
        with self.assertRaises(ValueError):
            input_validator.validate_under_over_sampling_input(class_populations=population,
                                                               minority_labels=minority_labels,
                                                               majority_labels=majority_labels,
                                                               base_minority=base_minority,
                                                               base_majority=base_majority)


if __name__ == '__main__':
    unittest.main()
