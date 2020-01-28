import os
from collections import deque


def extract_tagged_info(file_name: str, tag: str) -> str:
    """
    Extracts the incorporated info of an MVTS from the given file name. This is a generic method that
    considers the substring after `tag`, wrapped in square brackets, as the piece of information
    that is of interest. The value of the string `tag` depends on how the mvts files are named. Any
    piece of information can be encoded in the filename using a substring following the pattern
    as `t[x]`, where `t` is the `tag`, and `x` is the information of interest.

    For example, for the following filename:
        file_name: 'lab[B]1.0@1053_id[345]_st[2011-01-24T03:24:00]_et[2011-01-24T11:12:00].csv'

        tag : 'lab'

    The tag 'lab' can be used to extract the MVTS class 'B', or the tag `et` might be used to get
    access to the end time '2011-01-24T11:12:00', and so on.

    :param file_name: MVTS filename with a class-label string encoded in it that follows the
           description above.
    :param tag: A string that points to the piece of info of interest.

    :return: The embedded class label of the given filename.
    """
    sub_string: str = get_substring(file_name, tag)
    return sub_string


# -------------------------------------------
#              HELPER METHODS
# -------------------------------------------
def get_end_pair_index(s: str, i: int):
    """
    This method takes a string and an integer as inputs. It searches for the opening brace'[' using
    the given integer as the index of the opening brace in the given string. It outputs the index
    of the closing pair ']' in the string.

    :param s: The input string.
    :param i: Index position of the opening brace '['

    :return: Index position of the closing pair ']' of the opening brace '['
    """

    # If input is invalid.
    if s[i] != '[':
        return -1

    # Create a deque to use it as a stack.
    d = deque()

    # Traverse through all elements
    # starting from i.
    for k in range(i, len(s)):

        # Pop a starting bracket
        # for every closing bracket
        if s[k] == ']':
            d.popleft()

            # Push all starting brackets
        elif s[k] == '[':
            d.append(s[i])

            # If deque becomes empty
        if not d:
            return k

    return -1


def get_substring(file_name: str, left_expression: str) -> str:
    """
    Extracts the substring after the id_tag(given) and in between first occurrence of [ ]
    from the filename (given).
    Filename format: lab[B]1.0@1053_id[345]_st[2011-01-24T03:24:00]_et[2011-01-24T11:12:00].csv
    Left_expression : id
    Extracted Substring : 345

    :param file_name: A string that contains the time-series filename with the specified format
    :param left_expression: A string in the filename that points to the beginning of the embedded
                            substring.

    :return: The embedded substring of the given file name.
    """
    file_name = os.path.basename(os.path.normpath(file_name))
    start_index: int = file_name.find(left_expression) + len(left_expression)
    end_index = get_end_pair_index(file_name, start_index)

    if end_index == -1:
        raise Exception(
            """
            Filename format is incorrect. Pair of braces not found after given expression.
            """
        )

    return file_name[start_index + 1:end_index]
