import matplotlib as plt
import os
import datetime
from os import walk
import pandas as pd
import numpy as np
from tdigest import TDigest
from datetime import timedelta

import CONSTANTS as CONST

_5num_colnames: list = ['min', '25th', '50th', '75th', 'max']
_summary_keywords: dict = {"params_col": 'Feature-Name',
                           "null_col": "Null-Count",
                           "count_col": "Count",
                           "label_col": "Label"}


def run_tdigest_on_data(n_mvts, feature_list):
    path_to_dataset = os.path.join(CONST.ROOT, 'data/petdataset_01/')
    path_to_dataset, _, all_mvts_paths = next(walk(path_to_dataset))
    all_mvts_paths = all_mvts_paths[:n_mvts]
    total_param = len(feature_list)
    all_timediffs = []
    digests = [TDigest() for i in range(total_param)]

    i = 0
    for f in all_mvts_paths:
        if not f.endswith('.csv'):
            continue
        i += 1
        abs_path = os.path.join(path_to_dataset, f)
        df_mvts: pd.DataFrame = pd.read_csv(abs_path, sep='\t')
        df_req = pd.DataFrame(df_mvts[feature_list]).select_dtypes([np.number])
        j = 0
        for (param, series) in df_req.iteritems():
            series = np.array(series.values.flatten())

            s_time = datetime.datetime.now()  # -------- START TIME ------------
            digests[j].batch_update(series)
            digests[j].compress()
            e_time = datetime.datetime.now()  # -------- END TIME --------------
            delta = e_time - s_time

            all_timediffs.append(delta)
            j += 1

    return all_timediffs


def main():
    n = 10
    feature_list = ['TOTUSJH']
    all_timediffs = run_tdigest_on_data(n, feature_list)
    microseconds = [t.microseconds for t in all_timediffs]  # t.total_seconds()
    res_df = pd.DataFrame({'ITER': np.arange(len(microseconds)),
                           'MIC-SEC': microseconds
                           })
    res_df.to_csv('tDigest_incremental_exec_time.csv', sep=',', header=True, index=False,)


if __name__ == '__main__':
    main()
