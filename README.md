## MVTS Data Toolkit
### A Toolkit for Pre-processing Multivariate Time Series Data

* **Title:** MVTS Data Toolkit: A Toolkit for Pre-processing Multivariate Time Series Data
* **Journal:** SoftwareX Journal (Elsevier) -- [*under-review*]
* **Authors:** Azim Ahmadzadeh [>](https://www.azim-a.com/), Kankana Sinha [>](https://www.linkedin.com/in/kankana-sinha-4b4b13131/), Berkay Aydin [>](https://grid.cs.gsu.edu/~baydin2/), Rafal A. Angryk [>](https://grid.cs.gsu.edu/~rangryk/)
* **Demo Author:** Azim Ahmadzadeh
* **Last Modified:** Jan 24, 2020


![MVTS_Date_Toolkit Icon](https://bitbucket.org/gsudmlab/mvtsdata_toolkit/raw/c8f7e0edcfd899c93d9356d52b7ed8c6b500de04/__icon/MVTS_Data_Toolkit_icon2.png)


**Abstract:** We developed a domain-independent Python package to facilitate the
preprocessing routines required in preparation of any multi-class, multivariate time
series data. It provides a comprehensive set of 48 statistical features for extracting
the important characteristics of time series. The feature extraction process is
automated in a sequential and parallel fashion, and is supplemented with an extensive
summary report about the data. Using other modules, different data normalization
methods and imputation are at users' disposal. To cater the class-imbalance issue,
that is often intrinsic to real-world datasets, a set of generic but user-friendly,
sampling methods are also developed.


**This package provides:**

*  *Feature Collection:* A collection of 48 statistical features useful for analysis
of time series,
*  *Feature Extraction:* An automated feature-extraction process, with both parallel
and sequential execution capabilities,
*  *Visualization:* Quick and easy visualization for analysis of the extracted features, 
*  *Data Analysis:* A quick analysis of the mvts data and the extracted features, in
tabular and illustrative modes,
*  *Normalization:* A set of data transformation tools for normalization of the
extracted features,
*  *Sampling:* A set of generic methods to provide an array of undersampling and
oversampling remedies for balancing the class-imbalance datasets. 


----
[![PyPI license](https://img.shields.io/pypi/l/ansicolortags.svg??style=flat-square&logo=appveyor)](https://opensource.org/licenses/MIT)
[![PyPI license](https://img.shields.io/badge/PyPI-0.1-orange??style=flat-square&logo=appveyor)](https://pypi.org/project/mvtsdata-toolkit/)
[![PyPI license](https://img.shields.io/badge/Doc-Sphinx-blue??style=flat-square&logo=appveyor)](http://dmlab.cs.gsu.edu/docs/mvtsdata_toolkit/)
----
 
#### Requirements
*  Python > 3.6
*  For a list of all required packages, see [requirements.txt](./requirements.txt).

----
#### Try it online
Click on the badge below to try the demo provided in the notebook `demo.ipynb`, online:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/git/https%3A%2F%2Fbitbucket.org%2Fgsudmlab%2Fmvtsdata_toolkit%2Fsrc%2Fmaster/master?filepath=demo.ipynb)


----
#### Install it from PyPI
You can install this package, directly from Python Package Index (PyPI), using `pip` as follows:

```pip install mvtsdata-toolkit```

[![PyPI license](https://img.shields.io/badge/PyPI-0.1-orange??style=flat-square&logo=appveyor)](https://pypi.org/project/mvtsdata-toolkit/)

----
#### See Documentation
Check out the documentation of the project here:

[![PyPI license](https://img.shields.io/badge/Doc-Sphinx-blue??style=flat-square&logo=appveyor)](http://dmlab.cs.gsu.edu/docs/mvtsdata_toolkit/)
 
 
([http://dmlab.cs.gsu.edu/docs/mvtsdata_toolkit/](http://dmlab.cs.gsu.edu/docs/mvtsdata_toolkit/))


----

### Data Rules:

#### MVTS Files

It is assumed that the input dataset is a collection of multivariate time series (mvts), following
these assumptions:

1.  Each mvts is stored in a `tab`-delimited, csv file. Each column represents either the time
 series or some metadata such as timestamp. An mvts data with `t`
time series and `k` metadata columns, each of length `d`, has a dimension of
`d * (t + k)`).

2.  File names can also be used to have some metadata encoded using a *tag* followed by
 `[]`, for each piece of info. The *tag* indicates
what that piece of info is about, and the actual information should be stored inside
the proceeding square brackets. For instance, `A_id[123]_lab[1].csv` indicates that
this mvts is assigned the id `123` and the label `1`. If *tag*s are used, the
 metadata will be extracted and added to the extracted features automatically. To learn more
  about how the *tag*s can be used see 
the documentation in [features.feature_extractor.py](./features/feature_extractor.py)
.
  
3.  If the embedded values contain paired braces within `[]`, (e.g. for id,
`id[123[001]]`), then the metadata extractor would still be able to extract the info
correctly, however for unpaired braces (e.g. for id,
`id[123[001]`) it will raise an exception.

----
## Main Components:
*  All statistical features can be found in
[features.feature_collection](./features/feature_collection.py).
*  Code for parallel and sequential feature extraction can be found in
[features.feature_extractor](./features/feature_extractor.py).
*  Code for parallel and sequential analysis of raw mvts can be found in
[data_analysis.mvts_data_analysis](./data_analysis/mvts_data_analysis.py). 
*  Code for analysis of the extracted features can be found in
[data_analysis.extracted_features_analysis](./data_analysis/extracted_features_analysis.py).
*  Code for data normalization can be found in
[normalizing.normalizer](./normalizing/normalizer.py).
*  Code for sampling methods can be found in
[sampling.sampler](./sampling/sampler.py).


----

## Demo
The Jupyer notebook [demo](./demo.ipynb) is designed to give a tour of the
main functionalities of MVTS Data Toolkit. Users can either click on the
*binder* badge and run it online, or clone this project and run it on
their local machine.

A dataset of 2000 mvts files can be downloaded within the steps of this
demo. 

----
## Example Usage

In following examples, the string `'/PATH/TO/CONFIG.YML'` points to the
user's configuration file.
 
----
#### Data Analysis
This package allows analysis of both raw mvts data and the extracted
features.

Using [mvts_data_analysis](./data_analysis/mvts_data_analysis.py) module
users can easily get a glimpse of their raw data.

```python
from data_analysis.mvts_data_analysis import MVTSDataAnalysis
mda = MVTSDataAnalysis('/PATH/TO/CONFIG.YML')
mda.compute_summary(first_k=50,
                    params_name=['TOTUSJH', 'TOTBSQ', 'TOTPOT'])
```
Then, `mda.print_stat_of_directory()` gives the size of the data, in total
and on average, and `mda.summary` returns a dataframe with several
statistics on each of the time series. The statistics are `Val-Count`,
`Null-Count`, `mean`, `min`, `25th` (percentile), `50th` (= median),
`75th`, and `max`.

For large datasets, it is recommended to use the parallel version of this
method, as follows:
```python
mda.compute_summary_in_parallel(first_k=50,
                                n_jobs=4,
                                params_name=['TOTUSJH', 'TOTBSQ', 'TOTPOT'],)
```
which utilizes 4 processes to extract the summary statistics in parallel.
We explained in our paper in more details, about computing the statistics
in parallel.

Using [extracted_features_analysis](./data_analysis/extracted_features_analysis.py)
module users can also get some analysis from the extracted features (see Section
Feature Extraction). Suppose the dataframe of the extracted features is
loaded as a pandas dataframe into a variable called
`extracted_features_df`. Then,

```python
from data_analysis.extracted_features_analysis import ExtractedFeaturesAnalysis
efa = ExtractedFeaturesAnalysis(extracted_features_df, excluded_col=['id'])
efa.compute_summary()
```
that excludes the column `id` of the extracted features from the analysis.
After the summary is computed, the following methods can be used:
```python
efa.get_class_population(label='lab')
efa.get_missing_values()
efa.get_five_num_summary()
```

 
----
#### Feature Extraction

This snippet shows how [feature_extractor](./features/feature_extractor.py)
module can be used, for extracting 4 statistics (i.e., *min*, *max*, *median*, and *mean*),
from 3 time series parameteres (i.e., *TOTUSJH*, *TOTBSQ*, and *TOTPOT*).

```python
from features.feature_extractor import FeatureExtractor

fe = FeatureExtractor(path_to_config='/PATH/TO/CONFIG.YML')
fe.do_extraction(features_name=['get_min', 'get_max', 'get_median', 'get_mean'],
                 params_name=['TOTUSJH', 'TOTBSQ', 'TOTPOT'], first_k=50)
```
Note that user's configuration file must have the path to the raw mvts
using the key `PATH_TO_MVTS`.

To benefit from the parallel execution, do:
```python
fe.do_extraction_in_parallel(n_jobs=4,
                             features_index=[0, 1, 2, 3],
                             params_index=[0, 1, 2], first_k=50)
```
Here, for the sake of providing a richer example, we used `features_index`
and `params_index` instead of their names. This numeric mapping of features
and parameters makes it easier to deal with a numerous lengthy names.
These lists will be mapped to the list of parameters and features provided
in the user's configuration file, under the keys `MVTS_PARAMETERS` and
`STATISTICAL_FEATURES`, respectively.

In `FeatureExtractor` class, several plotting functionalities are
implemented that can be easily used as follows:

```python
params = ['TOTUSJH_median', 'TOTUSJH_mean', 'TOTBSQ_median', 'TOTBSQ_mean']
fe.plot_boxplot(params)
fe.plot_violinplot(params)
fe.plot_correlation_heatmap(params)
fe.plot_covariance_heatmap(params)
fe.plot_splom(params)
``` 

  
----
#### Sampling
After the statistical features are extracted from the mvts data, to remedy
the class-imbalance issue (if exists) a set of generic sampling methods
are provided in [sampler](./sampling/sampler.py) module.

```python
from sampling.sampler import Sampler

sampler = Sampler(extracted_features_df, label_col_name='lab')
sampler.sample(desired_populations={'N': 100, 'Y': 100})
```
which randomly samples 100 instances of the `N` class and 100 instances
of the `Y` class. If either of the classes does not have enough samples,
then after the entire samples are taken, the remaining needed instances
will be sampled with replacement. Depending on the provided populations,
this method could be an *undersampling* or an *oversampling* technique.

Users can use *ratio*s instead of *size* as follows:
```python
sampler.sample(desrired_ratios = {'N': 0.50, 'Y': -1})
```
which means take 50% of `N`-class instances, and *all* of `Y`-class
instances.

For other approaches, see the [/demo](./demo.ipynb).

 
----
#### Normalizing
The extracted features often require normalization. Using
[normalizer](./normalizing/normalizer.py) module, it can be easily
normalized as follows:

```python
from normalizing import normalizer
normalizer.zero_one_normalize()
df_normalized = normalizer.zero_one_normalize(extracted_features_df)
``` 
that again, `extracted_features_df` is assumed to be a pandas dataframe
of the extracted features.

In this module, the following four normalizers are provided:

*  zero_one_normalizer()
*  negativeone_one_normalize()
*  standardize()
*  robust_standardize()

 
----
Extra files:
*  [bitbucket-pipelines.yml](./bitbucket-pipelines.yml) is a configuration
file for pipelining the deployment steps before each release.
*  [CONSTANTS.py](./CONSTANTS.py) keeps track of the root directory, and
a few other pieces of information that are needed for the demo.
*  [demo.ipynb](./demo.ipynb) is the demo Jupyter notebook that can walk
the interested users through the functionalities this toolkit provides.
*  [README.md](./README.md) has the content of this very manual.
*  [requirements.txt](./requirements.txt) keeps track of all dependencies.
*  [setup.py](./setup.py) is used to generate the binary files needed for
generating the pip-installble version of this package.

----
#### Citation

Currently, this package is under review in [SoftwareX journal](https://www.journals.elsevier.com/softwarex). If you are interested in using this, I can share the manuscrip with you. Till it is published, it can be cited as follows:

```
@article{ahmadzadeh2020mvts,
  title={MVTS-Data Toolkit: A Python Package for Preprocessing Multivariate Time Series Data}},
  author={Azim Ahmadzadeh, Kankana Sinha, Berkay Aydin, Rafal A. Angryk},
  journal={SoftwareX},
  volume={},
  pages={},
  year={under-review},
  publisher={Elsevier}
}
```