import sys
import datetime
import pandas as pd
import numpy as np
from tdigest import TDigest
from pympler.asizeof import asizeof

'''
    This is written to keep track of execution time for tDigest's operations
    in each iteration.
'''


def run_tdigest(n_array, length_of_array):
    """
    this method generates `n_array` array of `length_of_array` random numbers, and runs tDigest
    over each of those arrays and record both the execution time (micro-seconds) and memory
    consumption (byte).

    :param n_array: number of array to be randomly generated.
    :param length_of_array: length of each randomly generated array.
    :return: a pair of dataframes, first consists of the recorded times and the second consists
    of the recorded memory.
    """

    all_timediffs = []
    all_mem_consumptions = []
    digest = TDigest()
    for x in range(n_array):
        s_time = datetime.datetime.now()  # -------- START TIME ------------
        digest.batch_update(np.random.random(length_of_array))
        digest.compress()
        e_time = datetime.datetime.now()  # -------- START TIME ------------
        mem_consumption = asizeof(digest)
        delta = e_time - s_time
        all_timediffs.append(delta)
        all_mem_consumptions.append(mem_consumption)

    return all_timediffs, all_mem_consumptions


def time_mem_test(n_iterations, n_array, length_of_array):
    """
    This is simply a wrapper for the method `run_tdigest` to repeat the experiment `n_iterations`
    times and concatenate the results in the form of a dataframe with `n_iterations` columns.
    The results will be stored in two different csv files.

    :param n_iterations:
    :param n_array: number of arrays to be randomly generated.
    :param length_of_array: length of the randomly-generated arrays.
    :return: None.
    """
    all_time_dfs = []
    all_mem_dfs = []
    for i in range(n_iterations):
        all_timediffs, all_mems = run_tdigest(n_array, length_of_array)

        microseconds = [t.microseconds for t in all_timediffs]  # t.total_seconds()
        times_df = pd.DataFrame(microseconds, columns=['Iter-{}'.format(i+1)])
        all_time_dfs.append(times_df)

        mems_df = pd.DataFrame(all_mems, columns=['Iter-{}'.format(i+1)])
        all_mem_dfs.append(mems_df)

        print('End of iteration {}.'.format(i+1))

    all_time_dfs = pd.concat(all_time_dfs, axis=1)
    out_csv_name = 'tDigest_time_test_i[{}]_n[{}]_l[{}].csv'.format(n_iterations, n_array,
                                                                    length_of_array)
    all_time_dfs.to_csv(out_csv_name, sep=',', header=True, index=False)

    all_mem_dfs = pd.concat(all_mem_dfs, axis=1)
    out_csv_name = 'tDigest_mem_test_i[{}]_n[{}]_l[{}].csv'.format(n_iterations, n_array,
                                                                   length_of_array)
    all_mem_dfs.to_csv(out_csv_name, sep=',', header=True, index=False)


def get_size(obj, seen=None):
    """
    Recursively finds size of objects
    Method source: [https://goshippo.com/blog/measure-real-size-any-python-object/]
    """
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


def main():
    n_iterations = 10
    n_array = 1000
    length_of_array = 100
    time_mem_test(n_iterations, n_array, length_of_array)


if __name__ == '__main__':
    main()
