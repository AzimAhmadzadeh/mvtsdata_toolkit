import pandas as pd
import numpy as np
import CONSTANTS as CONST
import os
import sys
import utils
from os import path, walk
from tdigest import TDigest
import matplotlib.pyplot as plt


class TestDistributed:

    def __init__(self):
        self.path_to_root = os.path.join('..', CONST.IN_PATH_TO_MVTS)
        self.path_to_dest = os.path.join('..', CONST.OUT_PATH_TO_EXTRACTED_FEATURES)
        self.out_file_name = 'distributed_stat.csv'
        self.df_all_features = pd.DataFrame()
        self.df = pd.DataFrame()

    def calculate_one(self, parameter):

        dirpath, _, all_csv_files = next(walk(CONST.IN_PATH_TO_MVTS))
        # all_csv_files.remove('.DS_Store')
        digest = TDigest()
        i = 0
        for f in all_csv_files:
            sys.stdout.flush()
            i += 1
            abs_path = os.path.join(dirpath, f)
            df_mvts: pd.DataFrame = pd.read_csv(abs_path, sep='\t')

            df_mvts = utils.interpolate_missing_vals(df_mvts)
            arrnd = np.array(df_mvts[[parameter]].values.flatten())
            digest.batch_update(arrnd)
            digest.compress()
            if i == 1:
                cumarr = df_mvts[parameter]
            else:

                cumarr = cumarr.append(df_mvts[parameter], ignore_index=False, verify_integrity=False)
        print(digest.to_dict())
        print(cumarr.size)
        print('TDigest 50 percentile:', digest.percentile(50))
        print('Manual percentile:', np.nanpercentile(cumarr, 50))

    def calculate_all(self, parameter_list: list = None):
        df = pd.DataFrame(columns=['Feature', 'E10', 'A10', 'E25', 'A25', 'E50', 'A50', 'E75', 'A75', 'E90', 'A90'])
        print(self.path_to_root)
        dirpath, _, all_csv_files = next(walk(CONST.IN_PATH_TO_MVTS))

        n = len(all_csv_files)
        total_param = parameter_list.__len__()
        param_seq = [str for i in range(total_param)]
        cum_array = [float for i in range(total_param)]
        digest = [TDigest() for i in range(total_param)]

        i = 0

        for f in all_csv_files:
            # Only .csv file needs to be processed
            if f.lower().find('.csv') != -1:
                sys.stdout.flush()
                i += 1
                abs_path = os.path.join(dirpath, f)

                df_mvts: pd.DataFrame = pd.read_csv(abs_path, sep='\t')

                df_mvts = utils.interpolate_missing_vals(df_mvts)
                # keep the requested params only
                df_req = pd.DataFrame(df_mvts[parameter_list], dtype=float)
                j = 0

                for (param, series) in df_req.iteritems():

                    series = np.array(df_req.iloc[:, [j]].values.flatten())
                    param_seq[j] = param
                    digest[j].batch_update(series)
                    digest[j].compress()
                    if i == 1:
                        cum_array[j] = df_req.iloc[:, [j]]
                    else:
                        cum_array[j] = np.append(cum_array[j],df_req.iloc[:, [j]])
                    j += 1


        k = 0
        for param in param_seq:
            df.loc[k] = [param, digest[k].percentile(10), np.nanpercentile(cum_array[k], 10), digest[k].percentile(25),
                         np.nanpercentile(cum_array[k], 25), digest[k].percentile(50),
                         np.nanpercentile(cum_array[k], 50),
                         digest[k].percentile(75), np.nanpercentile(cum_array[k], 75), digest[k].percentile(90),
                         np.nanpercentile(cum_array[k], 90)]
            k += 1

        pd.set_option('display.max_columns', 500)
        print(df)
        df.to_csv('test_distributed.csv',sep=',',index= 0)
        self.df = df

    def ploting_function(self):
        """

        """
        values = [float for i in range(2)]
        self.df = pd.read_csv('test_distributed.csv',sep=',')

        fig, (ax1, ax2) = plt.subplots(ncols = 2)
        names = ['10','25','50','75','90']
        values[0] = self.df[['A10','A25','A50','A75','A90']].loc[0]
        values[1] = self.df[['E10','E25','E50','E75','E90']].loc[0]
        ax1.plot(names,values[0],label='Actual')
        ax2.plot(names, values[1], label='Estimated')
        ax1.legend()
        ax2.legend()
        fig.text(0.5, 0.04, 'Percentile', ha='center')
        fig.text(0.04, 0.5, 'Value', va='center', rotation='vertical')
        fig.text(0.5, 0.96, 'TOTUSJH', ha='center')

        plt.show()


def main():
    import time
    from datetime import timedelta
    start_time = time.monotonic()

    ds = TestDistributed()
    parameter = 'TOTUSJH'
    ds.calculate_one(parameter)
    ds.calculate_all(CONST.CANDIDATE_PHYS_PARAMETERS)

    end_time = time.monotonic()
    print(timedelta(seconds=end_time - start_time))
    ds.ploting_function()


if __name__ == '__main__':
    main()
