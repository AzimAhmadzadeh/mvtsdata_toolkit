import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# '-g'  # solid green
# '--c' # dashed cyan
# '-.k' # dashdot black
# ':r'  # dotted red


def plot_iter_memory():
    """
    Loads the results in 'experiment_tdigest_time_memory.py' and plots
    a line plot of iteration vs time. Time corresponds to the execution
    time (in microseconds) recored in the experiment.
    """
    mem_df = pd.read_csv('./tDigest_mem_test_i[10]_n[1000]_l[100].csv')
    mean_mems = np.mean(mem_df, axis=1)
    x_vals = np.arange(len(mean_mems))

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Memory Consumption (Microseconds)')
    ax1.legend(loc='upper left', labels=['Memory (bytes)'])
    ax1.yaxis.grid(True)
    ax1.xaxis.grid(True)
    # ax.yaxis.grid(True, which='major')
    # ax.yaxis.grid(True, which='minor')
    plt.title('Experiment on Memory Consumption of tDigest')
    plt.plot(x_vals, mean_mems, color='c', linestyle='-', linewidth=2)
    fig.tight_layout()
    plt.show()
    # plt.savefig('a1.png')


def plot_iter_time():
    """
    Loads the results in 'experiment_tdigest_time_memory.py' and plots
    a line plot of iteration vs time. Time corresponds to the execution
    time (in microseconds) recored in the experiment.
    """
    times_df = pd.read_csv('./tDigest_time_test_i[10]_n[1000]_l[100].csv')
    mean_times = np.mean(times_df, axis=1)
    x_vals = np.arange(len(mean_times))

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Execution Time (Microseconds)')
    ax1.legend(loc='upper left', labels=['time (micro-seconds)'])
    ax1.yaxis.grid(True)
    ax1.xaxis.grid(True)
    # ax.yaxis.grid(True, which='major')
    # ax.yaxis.grid(True, which='minor')
    plt.title('Experiment on Execution Time of tDigest')
    plt.plot(x_vals, mean_times, color='c', linestyle='-', linewidth=2)
    fig.tight_layout()
    plt.show()
    # plt.savefig('a1.png')


if __name__=="__main__":
    plot_iter_time()
    plot_iter_memory()