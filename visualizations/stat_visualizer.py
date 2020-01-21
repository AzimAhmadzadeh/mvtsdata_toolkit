import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from normalizing import normalizer


class StatVisualizer:

    def __init__(self, path_to_extracted_features: str = None,
                 extracted_features: pd.DataFrame = None,
                 normalize: bool = True):
        """
        A constructor that loads the data and if required, normalizes the values into the [0,1]
        range.

        :param path_to_extracted_features: The absolute or relative path to the extracted features.
        :param extracted_features:
        :param normalize: False, if the ranges of values for different features are not far apart
                          and therefore, the boxplot does not need normalization. The default
                          value is True.
        """
        if path_to_extracted_features and extracted_features:
            raise ValueError(
                """
                Both of the arguments, `path_to_extracted_features` and `extracted_features`, 
                cannot be given at the same time!
                """
            )
        self.path_to_extracted_features = path_to_extracted_features
        self.extracted_features = extracted_features
        if self.path_to_extracted_features:
            self.__verify_path_to_extracted_features()
            self.extracted_features = pd.read_csv(self.path_to_extracted_features, sep='\t')
        if normalize:
            self.extracted_features = normalizer.zero_one_normalize(self.extracted_features)

    def __verify_path_to_extracted_features(self):
        """
        A private method that verifies whether the given path exist or not, and also whether it
        points to a csv file or not.

        :return: None
        """
        if not os.path.isfile(self.path_to_extracted_features):
            raise ValueError(
                """
                The given path (printed below) does not exist!
                \t{}
                """.format(self.path_to_extracted_features)
            )

        if not self.path_to_extracted_features.endswith('.csv'):
            raise ValueError(
                """
                The given file (printed below) is not a CSV file!
                \t{}
                """.format(self.path_to_extracted_features)
            )

    def boxplot_extracted_features(self, feature_names: list, output_path: str = None):
        """
        Generates a plot of boxplots, one for each extracted feature.

        :param feature_names: a list of feature-names indicating the columns of interest for this
                              visualization.
        :param output_path: If given, the generated plot will be stored instead of shown.
                            Otherwise, it will be only shown if the running environment
                            allows it.
        :return: None
        """
        # Plot from:
        # https: // seaborn.pydata.org / examples / horizontal_boxplot.html
        df = pd.melt(self.extracted_features[feature_names], var_name='feature', value_name='value')
        sns.set(style="ticks")

        fig, ax = plt.subplots(figsize=(10, len(feature_names)))

        # Plot the orbital period with horizontal boxes
        sns.boxplot(x="value", y="feature", data=df, whis="range", palette="vlag")

        # Add in points to show each observation
        sns.swarmplot(x="value", y="feature", data=df, size=2, color=".3", linewidth=0)

        # Tweak the visual presentation
        ax.xaxis.grid(True, which='major', linestyle='dotted', linewidth='0.5', color='gray')
        sns.despine(trim=True, left=True)

        # Hide these grid behind plot objects
        ax.set_axisbelow(True)
        ax.set_title('Boxplot of Extracted Features', fontsize=20)
        ax.set_ylabel('')
        ax.set_xlabel('Value', fontsize=12)

        if output_path:
            if not output_path.endswith('.png'):
                output_path += '.png'
            fig.savefig(output_path, dpi=200)
            plt.close(fig)
        else:
            plt.show()

    def plot_violinplot(self, feature_names: list, output_path: str = None):
        """
        Generates a set of violin-plots, one for each extracted feature.

        :param feature_names: a list of feature-names indicating the columns of interest for this
                              visualization.
        :param output_path: If given, the generated plot will be stored instead of shown.
                                Otherwise, it will be only shown if the running environment allows it.
        :return: None
        """
        df = self.extracted_features[feature_names]
        ax = sns.violinplot(data=df, orient='h', order=feature_names, palette="vlag")
        ax.set_title('Violin Plot of Extracted Features', fontsize=20)
        fig = ax.figure
        fig.set_size_inches((10, len(feature_names)))
        # fig.set_ticklabels(rotation=45, va="center")
        fig.tight_layout(pad=1)

        if output_path:
            if not output_path.endswith('.png'):
                output_path += '.png'
            fig.savefig(output_path, dpi=200)
            plt.close(fig)
        else:
            plt.show()

    def plot_splom(self, feature_names: list, output_path: str = None):
        """
        Generates a SPLOM, or a scatter plot matrix, for all pairs of features. Note that for a
        large number of features this may take a while (since each cell of the matrix is a
        scatter plot on its own), and also the final plot may become very large.

        :param feature_names: a list of feature-names indicating the columns of interest for this
                              visualization.
        :param output_path: If given, the generated plot will be stored instead of shown.
                                Otherwise, it will be only shown if the running environment
                                allows it.
        :return: None
        """
        df = self.extracted_features[feature_names]

        ax = sns.pairplot(df)
        # ax.set_title('SPLOM of Extracted Features', fontsize=20)
        fig = ax.fig
        fig.set_size_inches((len(feature_names)*2, len(feature_names)*2))
        fig.tight_layout(pad=1)

        if output_path:
            if not output_path.endswith('.png'):
                output_path += '.png'
            fig.savefig(output_path, dpi=200)
            plt.close(fig)
        else:
            plt.show()

    def plot_correlation_heatmap(self, feature_names: list, output_path: str = None):
        """
        Generates a heat-map for the correlation matrix of all pairs of given features.

        Note: Regardless of the range of correlations, the color-map is fixed to [-1, 1]. This is
        especially important to avoid mapping insignificant changes of values into significant
        changes of colors.

        :param feature_names: a list of feature-names indicating the columns of interest for this
                              visualization.
        :param output_path: If given, the generated plot will be stored instead of shown.
                                Otherwise, it will be only shown if the running environment
                                allows it.
        :return: None
        """
        df = self.extracted_features[feature_names]

        ax = sns.heatmap(df.corr(), cmap="vlag", vmin=-1, vmax=1)
        fig = ax.figure

        if len(feature_names) < 4:
            ax.set_title('Heat-map of Correlation Matrix', fontsize=10)
            fig.set_size_inches(4, 4)
        else:
            ax.set_title('Heat-map of Correlation Matrix', fontsize=20)
            fig.set_size_inches(len(feature_names), len(feature_names))
        fig.tight_layout(pad=1)

        plt.yticks(va="center")  # rotation=45, fontsize="10",

        if output_path:
            if not output_path.endswith('.png'):
                output_path += '.png'
            fig.savefig(output_path, dpi=200)
            plt.close(fig)
        else:
            plt.show()

    def plot_covariance_heatmap(self, feature_names: list, output_path: str = None):
        """
        Generates a heat-map for the covariance matrix of all pairs of given features.

        Note that covariance is not a standardized statistic, and because of this, the color-map
        might be confusing; when the difference between the largest and smallest covariance is
        insignificant, the colors may imply a significant difference. To avoid this, the values
        mapped to the colors (as shown next to the color-map) must be carefully taken into
        account in analysis of the covariance.

        :param feature_names: a list of feature-names indicating the columns of interest for this
                              visualization.
        :param output_path: If given, the generated plot will be stored instead of shown.
                                Otherwise, it will be only shown if the running environment
                                allows it.
        :return: None
        """
        df = self.extracted_features[feature_names]

        # Heatmap of covariance matrix
        ax = sns.heatmap(df.cov(), cmap="vlag")
        ax.set_title('Heat-map of Covariance Matrix')
        fig = ax.figure

        if len(feature_names) < 4:
            ax.set_title('Heat-map of Covariance Matrix', fontsize=10)
            fig.set_size_inches(4, 4)
        else:
            ax.set_title('Heat-map of Covariance Matrix', fontsize=20)
            fig.set_size_inches(len(feature_names), len(feature_names))
        fig.tight_layout(pad=1)

        plt.yticks(va="center")  # rotation=45, fontsize="10",

        if output_path:
            if not output_path.endswith('.png'):
                output_path += '.png'
            fig.savefig(output_path, dpi=200)
            plt.close(fig)
        else:
            plt.show()


def main():
    path_to_df = '/home/azim/CODES/PyWorkspace/mvtsdata_toolkit/data/extracted_features' \
                 '/extracted_features_parallel_3_pararams_4_features.csv'
    params = ['TOTUSJH_min', 'TOTUSJH_max',
              'TOTUSJH_median', 'TOTUSJH_mean', 'TOTBSQ_min',
              'TOTBSQ_max', 'TOTBSQ_median', 'TOTBSQ_mean', 'TOTPOT_min']
    vis = StatVisualizer(path_to_df)
    vis.boxplot_extracted_features(params)
    vis.plot_violinplot(params)
    vis.plot_splom(params)
    vis.plot_correlation_heatmap(params)
    vis.plot_covariance_heatmap(params)


if __name__ == "__main__":
    main()
