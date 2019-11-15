import matplotlib as plt
import os
import time
import sys
from os import path, makedirs, walk
import pandas as pd
import numpy as np
from tdigest import TDigest
from hurry.filesize import size

import CONSTANTS as CONST

_5num_colnames: list = ['min', '25th', '50th', '75th', 'max']
_summary_keywords: dict = {"params_col": 'Feature-Name',
                           "null_col": "Null-Count",
                           "count_col": "Count",
                           "label_col": "Label"}


def run_tdigest_on_data():
    path_to_dataset = os.path.join(CONST.ROOT, 'data/petdataset_01/')
    path_to_dataset, _, all_mvts_paths = next(walk(path_to_dataset))
    all_mvts_paths = all_mvts_paths[:10]
    feature_list = ['TOTUSJH', 'TOTBSQ', 'TOTPOT', 'TOTUSJZ']
    total_param = len(feature_list)
    param_seq = [""] * total_param
    digests = [TDigest() for i in range(total_param)]

    i = 0
    j = 0
    for f in all_mvts_paths:
        if not f.endswith('.csv'):
            continue
        i += 1
        abs_path = os.path.join(path_to_dataset, f)
        df_mvts: pd.DataFrame = pd.read_csv(abs_path, sep='\t')
        df_req = pd.DataFrame(df_mvts[feature_list]).select_dtypes([np.number])

        j = 0
        for (param, series) in df_req.iteritems():
            if not series.empty:
                series = np.array(series.values.flatten())
                param_seq[j] = param
                digests[j].batch_update(series)
                digests[j].compress()

            j += 1
    all_columns = _5num_colnames[:]
    all_columns.insert(0, _summary_keywords['params_col'])
    summary_stat_df = pd.DataFrame(columns=all_columns)

    for i in range(total_param):
        attname = param_seq[i]
        col_min = col_q1 = col_mean = col_q3 = col_max = 0
        if digests[i]:
            col_min = digests[i].percentile(0)
            col_q1 = digests[i].percentile(25)
            col_mean = digests[i].percentile(50)
            col_q3 = digests[i].percentile(75)
            col_max = digests[i].percentile(100)

        summary_stat_df.loc[i] = [attname, col_min, col_q1, col_mean, col_q3, col_max]

    summary_stat_df.reset_index(inplace=True)
    summary_stat_df.drop(labels='index', inplace=True, axis=1)


def ploting_function(self):
    values = [0.0 for i in range(2)]
    self.df = pd.read_csv('test_distributed.csv', sep=',')

    fig, (ax1, ax2) = plt.subplots(ncols=2)
    names = ['10', '25', '50', '75', '90']
    values[0] = self.df[['A10', 'A25', 'A50', 'A75', 'A90']].loc[0]
    values[1] = self.df[['E10', 'E25', 'E50', 'E75', 'E90']].loc[0]
    ax1.plot(names, values[0], label='Actual')
    ax2.plot(names, values[1], label='Estimated')
    ax1.legend()
    ax2.legend()
    fig.text(0.5, 0.04, 'Percentile', ha='center')
    fig.text(0.04, 0.5, 'Value', va='center', rotation='vertical')
    fig.text(0.5, 0.96, 'TOTUSJH', ha='center')

    plt.show()


def main():
    run_tdigest_on_data()
    # import time
    # from datetime import timedelta
    # start_time = time.monotonic()
    #
    # ds = TestDistributed()
    # parameter = 'TOTUSJH'
    # ds.calculate_one(parameter)
    # parameters = ['TOTUSJH', 'TOTBSQ', 'TOTPOT', 'TOTUSJZ']
    # ds.calculate_all(parameters)
    #
    # end_time = time.monotonic()
    # print(timedelta(seconds=end_time - start_time))
    # ds.ploting_function()


if __name__ == '__main__':
    main()
