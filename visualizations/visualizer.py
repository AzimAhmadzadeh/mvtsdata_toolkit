import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from normalizing import normalizer


def __verify_path_to_extracted_features(path_to_extracted_features: str):
    """
    A private method that verifies whether the given path exist or not, and also whether it
    points to a csv file or not.

    :return: None
    """
    if not os.path.isfile(path_to_extracted_features):
        raise ValueError(
            """
            The given path (printed below) does not exist!
            \t{}
            """.format(path_to_extracted_features)
        )

    if not path_to_extracted_features.endswith('.csv'):
        raise ValueError(
            """
            The given file (printed below) is not a CSV file!
            \t{}
            """.format(path_to_extracted_features)
        )


def boxplot_extracted_features(self, feature_names: list, output_filename: str = None):
    """
    Generates a list of boxplots, one for each extracted feature.

    :param feature_names: a list of feature-names indicating the columns of interest for this
                          visualization.
    :param output_filename: If given, the generated plot will be stored instead of shown.
                            Otherwise, it will be only shown if the running environment
                            allows it.
    :return: None
    """
    # Plot from:
    # https: // seaborn.pydata.org / examples / horizontal_boxplot.html
    df = pd.melt(self.df[feature_names], var_name='feature', value_name='value')
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

    if output_filename:
        if not output_filename.endswith('.png'):
            output_filename += '.png'
        fig.savefig(output_filename, dpi=200)
        plt.close(fig)
    else:
        plt.show()
