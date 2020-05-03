## MVTS Data Toolkit v0.2.6
### A Toolkit for Pre-processing Multivariate Time Series Data

* **Title:** MVTS Data Toolkit: A Toolkit for Pre-processing Multivariate Time Series Data
* **Journal:** SoftwareX Journal [>](https://www.journals.elsevier.com/softwarex) (Elsevier) -- [*under-review*]
* **Authors:** Azim Ahmadzadeh [>](https://www.azim-a.com/), Kankana Sinha [>](https://www.linkedin.com/in/kankana-sinha-4b4b13131/), Berkay Aydin [>](https://grid.cs.gsu.edu/~baydin2/), Rafal A. Angryk [>](https://grid.cs.gsu.edu/~rangryk/)
* **Demo Author:** Azim Ahmadzadeh
* **Last Modified:** May 03, 2020

![MVTS_Date_Toolkit Icon](https://bitbucket.org/gsudmlab/mvtsdata_toolkit/raw/c8f7e0edcfd899c93d9356d52b7ed8c6b500de04/__icon/MVTS_Data_Toolkit_icon2.png)


**Abstract:** We developed a domain-independent Python package to facilitate the
preprocessing routines required in preparation of any multi-class, multivariate time
series data. It provides a comprehensive set of 48 statistical features for extracting
the important characteristics of time series. The feature extraction process is
automated in a sequential and parallel fashion, and is supplemented with an extensive
summary report about the data. Using other modules, different data normalization
methods and imputations are at users' disposal. To cater the class-imbalance issue,
that is often intrinsic to real-world datasets, a set of generic but user-friendly,
sampling methods are also developed.


**This package provides:**

*  *Feature Collection:* A collection of 48 statistical features for analysis
of time series,
*  *Feature Extraction:* An automated feature-extraction process, with both parallel
and sequential execution capabilities,
*  *Visualization:* Several quick and easy visualization methods for analysis of the extracted
 features, 
*  *Data Analysis:* A quick analysis of the mvts data and the extracted features,
*  *Normalization:* A set of data transformation tools for normalization of the
extracted features,
*  *Sampling:* A set of generic methods to provide an array of undersampling and
oversampling remedies for balancing the class-imbalance datasets. 


----
[![PyPI license](https://img.shields.io/pypi/l/ansicolortags.svg??style=flat-square&logo=appveyor)](https://opensource.org/licenses/MIT)
[![PyPI license](https://img.shields.io/badge/PyPI-0.2.6-orange??style=flat-square&logo=appveyor)](https://pypi.org/project/mvtsdatatoolkit/)
[![PyPI license](https://img.shields.io/badge/Doc-Sphinx-blue??style=flat-square&logo=appveyor)](http://dmlab.cs.gsu.edu/docs/mvtsdata_toolkit/)
----
 
#### Requirements
*  Python >= 3.6
*  For a list of all required packages, see [requirements.txt](./requirements.txt).

----
#### Try it online
Click on the badge below to try the demo provided in the notebook `demo.ipynb`, online:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/git/https%3A%2F%2Fbitbucket.org%2Fgsudmlab%2Fmvtsdata_toolkit%2Fsrc%2Fmaster/master?filepath=.%2Fdemo.ipynb)
----
#### Install it from PyPI
You can install this package, directly from Python Package Index (PyPI), using `pip` as follows:
* Linux/Mac OS:

    ```pip install mvtsdatatoolkit```
* Windows:

**Note**: On windows, the *Microsooft Visual C++* must be
updated. Otherwise the error `Microsoft Visual C++ 14.0 is required`
might terminate the installation. To solve this issue, see
this [Medium post](https://medium.com/@jacky_ttt/day060-fix-error-microsoft-visual-c-14-0-is-required-629413e798cd)
that elaborates on this short [Stackoverflow answer](https://stackoverflow.com/a/40888720). 


[![PyPI license](https://img.shields.io/badge/PyPI-0.2.6-orange??style=flat-square&logo=appveyor)](https://pypi.org/project/mvtsdatatoolkit/)

----
#### See Documentation
Check out the documentation of the project here:

[![PyPI license](https://img.shields.io/badge/Doc-Sphinx-blue??style=flat-square&logo=appveyor)](http://dmlab.cs.gsu.edu/docs/mvtsdata_toolkit/)
  
----

### Data Rules:

#### MVTS Files

It is assumed that the input dataset is a collection of multivariate time series (mvts), following
these assumptions:

1.  Each mvts is stored in a `tab`-delimited, csv file. Each column represents either the time
 series or some metadata such as timestamp. An mvts dataset with `t`
time series and `k` metadata columns, each of length `d`, has a dimension of
`d * (t + k)`.

2.  File names can also be used to have some metadata encoded using a *tag* followed by
 `[]`, for each piece of info. The *tag* can be any string of characters and indicates
what that piece of info is about, and the actual information should be stored inside
the proceeding square brackets. For example, the file-name `A_id[123]_lab[1].csv`
indicates that this mvts is assigned the id `123` and the label `1`. If *tag*s are used,
during the feature extraction process, the metadata will be extracted and also added
to the tabular extracted features automatically. To learn more about how the *tag*s can
be used see the documentation in [features.feature_extractor.py](mvtsdatatoolkit/features/feature_extractor.py)
.
  
3.  If the embedded values contain paired braces within `[]`, (e.g. for id,
`id[123[001]]`), then the metadata extractor would still be able to extract the info
correctly, however for unpaired braces (e.g. for id,
`id[123[001]`) it will raise an exception.

----
## Main Components:
*  All statistical features can be found in
[features.feature_collection](mvtsdatatoolkit/features/feature_collection.py).
*  Code for parallel and sequential feature extraction process can be found in
[features.feature_extractor](mvtsdatatoolkit/features/feature_extractor.py).
*  Code for parallel and sequential analysis of raw mvts can be found in
[data_analysis.mvts_data_analysis](mvtsdatatoolkit/data_analysis/mvts_data_analysis.py). 
*  Code for analysis of the extracted features can be found in
[data_analysis.extracted_features_analysis](mvtsdatatoolkit/data_analysis/extracted_features_analysis.py).
*  Code for data normalization can be found in
[normalizing.normalizer](mvtsdatatoolkit/normalizing/normalizer.py).
*  Code for sampling methods can be found in
[sampling.sampler](mvtsdatatoolkit/sampling/sampler.py).


----

## Demo

A Jupyer notebook is provided to give a tour of the main
functionalities of the package. Running the demo is fairly
simple. You need the notebook and the example input.

#### 1. Notebook
The Jupyer notebook [demo](demo.ipynb) is at the root directory.

Users can try the demo in one of the three ways listed below:

* Online: click on the *binder* badge (see above) and you will be
able to follow the demo on a remote server online. This is the
simplest way to try the demo. A user would only need access to
the Internet for this method.
* Locally with package: `pip` install the `mvtsdatatoolkit` package on your local machine
and download and run the nodebook from the same (virtual or physical)
machine. (See the next section for more details.)
* Locally with source: Clone the `mvtsdata_toolkit` project, install the dependencies
(listed in [requirements.txt](./requirements.txt) and run the notebook from the same
(virtual or physical) machine.

#### 2. Input
A dataset of 2000 mvts files and a configuration file specifically defined for this
dataset will be downloaded along the steps of this demo.

The provided dataset is a subset of the benchmark dataset
called *Space Weather ANalytics for Solar Flares*
(*SWAN-SF*) [2] .

----
## Need Help Running Demo Locally?
Follow the steps below to run the demo notebook on your
local machine using *virtualenv* and without having to
clone the project. If you are more comfortable with
*conda*/*anaconda* make appropriate adjustments.

(Commands below are specific to Unix-base systems)

* Create a new directory and `cd` into it:
```bash
mkdir mvts_demo
cd mvts_demo/
```
* Inside `mvts_demo` directory create a new *virtualenv*
called `venv` and activate the virtual environment:
```bash
virtualenv -p /usr/bin/python3.6 venv
source venv/bin/activate
```
* Install `mvtsdatatoolkit` (this will consequently install
`notebook` library among other required libraries):
```bash
pip install mvtsdatatoolkit
``` 
* Download the notebook and start the Jupyter notebook:
```bash
wget https://bitbucket.org/gsudmlab/mvtsdata_toolkit/downloads/demo.ipynb
jupyter notebook
```
----

## Example Code Snippets

In following examples, the string `'/PATH/TO/CONFIG.YML'` 
points to the user's configuration file.
 
----
#### Data Analysis
This package allows analysis of both raw mvts data and the
extracted features.

Using [mvts_data_analysis](mvtsdatatoolkit/data_analysis/mvts_data_analysis.py) module
users can easily get a glimpse of their raw data.

```python
from mvtsdatatoolkit.data_analysis import MVTSDataAnalysis
mda = MVTSDataAnalysis('/PATH/TO/CONFIG.YML')
mda.compute_summary(first_k=50,
                    params_name=['TOTUSJH', 'TOTBSQ', 'TOTPOT'])
```
Then, `mda.print_stat_of_directory()` gives the size of the data, in total
and on average, and `mda.summary` returns a dataframe with several
statistics on each of the time series. The statistics are `Val-Count`,
`Null-Count`, `mean`, `min`, `max`, and the quartiles `25th`, `50th` (= median),
`75th`.

For large datasets, it is recommended to use the parallel version of this
method, as follows:
```python
mda.compute_summary_in_parallel(first_k=50,
                                n_jobs=4,
                                params_name=['TOTUSJH', 'TOTBSQ', 'TOTPOT'],)
```
which utilizes 4 processes to extract the summary statistics
in parallel, on the first `50` mvts files. For more details about
the parallel computation see the paper [1].

Using [extracted_features_analysis](mvtsdatatoolkit/data_analysis/extracted_features_analysis.py)
module users can also get some analyses from the extracted
features (see Section Feature Extraction). Suppose the
dataframe of the extracted features is loaded as a pandas
dataframe into a variable called `extracted_features_df`.
Then,

```python
from mvtsdatatoolkit.data_analysis import ExtractedFeaturesAnalysis
efa = ExtractedFeaturesAnalysis(extracted_features_df, excluded_col=['id'])
efa.compute_summary()
```
that excludes the column `id` of the extracted features from
the analysis and computes a set of summary statistics on all
extracted features.

After the summary is computed, the following methods can be used:
```python
efa.get_class_population(label='lab')
efa.get_missing_values()
efa.get_five_num_summary()
```

 
----
#### Feature Extraction

This snippet shows how [feature_extractor](mvtsdatatoolkit/features/feature_extractor.py)
module can be used, for extracting 4 statistics (i.e., *min*,
*max*, *median*, and *mean*), from 3 time series parameteres
(i.e., *TOTUSJH*, *TOTBSQ*, and *TOTPOT*) available in the
provided dataset.

```python
from mvtsdatatoolkit.features import FeatureExtractor

fe = FeatureExtractor(path_to_config='/PATH/TO/CONFIG.YML')
fe.do_extraction(features_name=['get_min', 'get_max', 'get_median', 'get_mean'],
                 params_name=['TOTUSJH', 'TOTBSQ', 'TOTPOT'], first_k=50)
```
Note that user's configuration file must contain the path
to the raw mvts using the key `PATH_TO_MVTS`.

To benefit from the parallel execution, do:
```python
fe.do_extraction_in_parallel(n_jobs=4,
                             features_index=[0, 1, 2, 3],
                             params_index=[0, 1, 2], first_k=50)
```
Here, for the sake of providing a richer example, we used
`features_index` and `params_index` instead of their names (that
was already shown in the previous example). This numeric mapping
of features and parameters makes it easier to deal with a
long array of lengthy names. These two lists will be mapped to
the list of parameters and features provided in the user's
configuration file, under the keys `MVTS_PARAMETERS` and
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
After the statistical features are extracted from the mvts data,
to remedy the class-imbalance issue (if exists) a set of generic
sampling methods are provided in [sampler](mvtsdatatoolkit/sampling/sampler.py)
module.

```python
from mvtsdatatoolkit.sampling.sampler import Sampler

sampler = Sampler(extracted_features_df, label_col_name='lab')
sampler.sample(desired_populations={'N': 100, 'Y': 100})
```
Assumming that the dataset has the class labels `Y` and `N`, this
snippet of code randomly samples 100 instances of the `N` class
and 100 instances of the `Y` class instances. If either of the
classes does not have enough samples, then after the entire
samples are taken, the remaining needed instances will be
sampled randomly with replacement. Depending on the provided
populations, this method could turn into an *undersampling* or
*oversampling* function.

Users can use *ratio* instead of *size* as follows:
```python
sampler.sample(desrired_ratios = {'N': 0.50, 'Y': -1})
```
which means take 50% of the entire population would be sampled
from `N`-class instances, and *all* of `Y`-class instances will
also be passed to the sampled data.

For other approaches, see the [/demo](demo1.ipynb).

 
----
#### Normalizing
The extracted features often require normalization. Using
the module [normalizer](mvtsdatatoolkit/normalizing/normalizer.py)
, the features can be quickly normalized as follows:

```python

from mvtsdatatoolkit.normalizing import normalizer
normalizer.zero_one_normalize()
df_normalized = normalizer.zero_one_normalize(extracted_features_df)
``` 
Again, `extracted_features_df` is assumed to be a pandas
dataframe of the extracted features.

In this module, the following four normalizers are implemented
on top of the *scikit-learn* library.

*  zero_one_normalizer()
*  negativeone_one_normalize()
*  standardize()
*  robust_standardize()

 
----
Extra files:
*  [bitbucket-pipelines.yml](./bitbucket-pipelines.yml) is a
configuration file for pipelining the deployment steps before
each release.
*  [CONSTANTS.py](./CONSTANTS.py) keeps track of some constant
variables such as root path.
*  [demo.ipynb](demo.ipynb) is the demo Jupyter notebook that
can walk the interested users through the functionalities this
toolkit provides.
*  [README.md](./README.md) has the content of this very manual.
*  [requirements.txt](./requirements.txt) keeps track of all
dependencies.
*  [setup.py](./setup.py) is used to generate the binary files
needed for generating the pip-installble version of this package.

----
#### Citation

Currently, this package is under review in
[SoftwareX journal](https://www.journals.elsevier.com/softwarex).
If you are interested in using this, I can share the manuscript with
you. Till it is published, it can be cited as follows:

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

----
#### References
[1] A. Ahmadzadeh, K. Sinha, 2020. "MVTS-Data Toolkit:
A Python Package for Preprocessing Multivariate Time
Series Data", (under review 2020))

[2] Angryk, R.A., Martens, P.C., Aydin, B., Kempton, D.,
Mahajan, S.S., Basodi, S., Ahmadzadeh, A., Cai, X.,
Boubrahimi, S.F., Hamdi, S.M., Schuh, M.A. and
Georgoulis, M.K., 2019. "Multivariate Time Series
Dataset for Space Weather Data Analytics".
Sci. Data, Nature, submitted (2019).
