## Multivariate Time Series Data Toolkit

* **Software Name:** MVTS Data Toolkit
* **Journal:** SoftwareX (Elsevier) -- [*submitted*]
* **Title:** MVTS-Data Toolkit: A Python Package for Preprocessing Multivariate Time Series Data
* **Authors:** Azim Ahmadzadeh, Kankana Sinha, Berkay Aydin, and Rafal Angryk
* **Submission Date:** Feb 2020


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
* *Feature Collection:* A collection of 48 statistical features useful for analysis
of time series.
* *Feature Extraction:* An automated feature-extraction process, with both parallel
and sequential execution capabilities.
* *Data Analysis:* A quick analysis of the mvts data and the extracted features, in
tabular and illustrative modes.
* *Normalization:* A set of data transformation tools for normalization of the
extracted features.
* *Sampling:* A set of generic methods to provide an array of undersampling and
oversampling remedies for balancing the class-imbalance datasets. 


----
[![PyPI license](https://img.shields.io/pypi/l/ansicolortags.svg??style=flat-square&logo=appveyor)](https://opensource.org/licenses/MIT)
[![PyPI license](https://img.shields.io/badge/PyPI-0.1-orange??style=flat-square&logo=appveyor)](https://pypi.org/project/mvtsdata-toolkit/)
[![PyPI license](https://img.shields.io/badge/Doc-Sphinx-blue??style=flat-square&logo=appveyor)](http://dmlab.cs.gsu.edu/docs/mvtsdata_toolkit/)
----
 
#### Requirements
* Python > 3.6
* For a list of all required packages, see [requirements.txt](./requirements.txt).

----
#### Try it online
Click on the badge below to try the demo provided in the notebook `demo.ipynb`, online:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/git/https%3A%2F%2Fbitbucket.org%2Fgsudmlab%2Fmvtsdata_toolkit%2Fsrc%2Fmaster/master?filepath=demo.ipynb)


----
#### Install it from PyPI
You can install this package, directly from Python Package Index (PyPI), using `pip` as follows:

```pip install mvtsdata-toolkit```

[![PyPI license](https://img.shields.io/badge/PyPI-0.1-orange??style=flat-square&logo=appveyor)](https://pypi.org/project/mvts-data-toolkit/)


----
#### See Documentation
Check out the documentation of the project here:

[![PyPI license](https://img.shields.io/badge/Doc-Sphinx-blue??style=flat-square&logo=appveyor)](http://dmlab.cs.gsu.edu/docs/mvtsdata_toolkit/)
 
 
([http://dmlab.cs.gsu.edu/docs/mvtsdata_toolkit/](http://dmlab.cs.gsu.edu/docs/mvtsdata_toolkit/))


----

### Data Rules:

#### MVTS Files

It is assumed that the input dataset is a collection of multivariate time series (mvts), following these assumptions:

1. Each mvts is stored in a `tab`-delimited, csv file. Each column represents either the time
 series or some metadata such as timestamp. An mvts data with `t`
time series and `k` metadata columns, each of length `d`, has a dimension of
`d * (t + k)`).

2. File names can also be used to have some metadata encoded using a *tag* followed by
 `[]`, for each piece of info. The *tag* indicates
what that piece of info is about, and the actual information should be stored inside
the proceeding square brackets. For instance, `A_id[123]_lab[1].csv` indicates that
this mvts is assigned the id `123` and the label `1`. If *tag*s are used, the
 metadata will be extracted and added to the extracted features automatically. To learn more
  about how the *tag*s can be used see 
the documentation in [features.feature_extractor.py](./features/feature_extractor.py)
.
  
3. If the embedded values contain paired braces within `[]`, (e.g. for id,
`id[123[001]]`), then the metadata extractor would still be able to extract the info
correctly, however for unpaired braces (e.g. for id,
`id[123[001]`) it will raise an exception.

----
## Main Components:
* All statistical features can be found in
[features.feature_collection](./features/feature_collection.py).
* Code for parallel and sequential feature extraction can be found in
[features.feature_extractor](./features/feature_extractor.py).
* Code for parallel and sequential analysis of raw mvts can be found in
[data_analysis.mvts_data_analysis](./data_analysis/mvts_data_analysis.py). 
* Code for analysis of the extracted features can be found in
[data_analysis.extracted_features_analysis](./data_analysis/extracted_features_analysis.py).
* Code for data normalization can be found in
[normalizing.normalizer](./normalizing/normalizer.py).
* Code for sampling methods can be found in
[sampling.sampler](./sampling/sampler.py).


----


## Example Usage

In the examples below, the string `'/PATH/TO/CONFIG.YML'` points to the
user's configuration file.
 
#### Data Analysis
This package allows analysis of both raw mvts data and the extracted
features.

Using [mvts_data_analysis](./data_analysis/mvts_data_analysis.py)
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
users can also get some analysis from the extracted features (see Section
Feature Extraction). Suppose the dataframe of the extracted features is
loaded as a pandas dataframe into a variable called
`extracted_features_df`. Then,

```python
from data_analysis.extracted_features_analysis import ExtractedFeaturesAnalysis
efa = ExtractedFeaturesAnalysis(extracted_features_df, excluded_col=['id'])
efa.compute_summary()
```
that excludes the column `id` of the extracted features from the analysis.

#### Feature Extraction

This snippet shows how [feature_extractor](./features/feature_extractor.py)
can be used, for extracting 4 statistics (i.e., *min*, *max*, *median*, and *mean*),
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













## Multivariate Timeseries Feature Extraction 
This module aims to simplify the feature extraction process for those interested in working with MVTS Data. The main objectives of this module are as follows.

This module provides:

1. a set of useful statistical features to be extracted form the multivariate time series, in order to transform the time series dataset into tabular data.
2. means to facilitate extraction of the features from a high dimensional multivariate time series dataset. 
3. several sampling methodologies tailored specifically for class-imbalance issue which is intrinsic to many MVTS dataset.
4. multiple normalization methods that can be used and potentially improve a forecast model's performance.
5. several performance metrics that several studies have shown to be effective in reflecting forecast models' performance.

In the following, we briefly introduce the dataset, all the means provided by this module, and some snippets of code to showcase how the module can be used.


### FEATURES PACKAGE [[features](./features)]
This package contains the statistical features that can be extracted from the time series, and the scripts needed to compute the features on the multivariate time series data.

#### Features Collection [[features.feature_collection.py](./features/feature_collection.py)]


#### Features Extractor (sequential) [[features.feature_extractor.py](./features/feature_extractor.py)]
This extracts statistical features from the multivariate time series dataset. Using this script, any of the features mentioned above can be selected to be computed on any subset of the physical parameters in SWAN.

The snippet below shows how to extract specified statistical features from specified physical parameters (declared in [CONSTANTS.py](CONSTANTS.py) file). The extracted features will be a `pandas.DataFrame` (stored as a csv file) with its shape being `N X M`, where `N` is the number of multivaraite time series available in `/path/to/FL/`, and `M` is the dimensionality of each extracted vector. In this exmaple, `M` equals the number of generated features (3*4), plus 4; four pieces of meta data, `NOAA_AR_NO`, `LABEL`, START_TIME`, and `END_TIME`.

```python
import os
import CONSTANTS as CONST
import features.feature_collection as fc
from features.feature_extractor import FeatureExtractor

# Input and output path
path_to_root = os.path.join('..', CONST.IN_PATH_TO_MVTS_FL)
path_to_dest = os.path.join('..', CONST.OUT_PATH_TO_RAW_FEATURES)
output_filename = 'extracted_features.csv'

# Prepare two lists, one for the statistical features and another for the physical parameters
stat_features = CONST.CANDIDATE_STAT_FEATURES
phys_parameters = CONST.CANDIDATE_PHYS_PARAMETERS

pc = FeatureExtractor(path_to_root, path_to_dest, output_filename)
pc.do_extraction(features_list=stat_features, params_name_list=phys_parameters)

```

### UTIL PACKAGE [[utils](./utils)]
This package serves five different functionality as helper methods to the overall tool:
#### MVTS Data Analysis[[utils.mvts_data_analysis.py](./data_analysis/mvts_data_analysis.py)]
This class takes the folder location(path) of the MVTS dataset in time of instance creation. By calling method compute_summary()
it extracts each mvts(.csv) file and computes the statistical parameters of each numeric physical feature that belongs to the MVTS.
TDigest module is used in order to calculate the statistical parameters(like percentile) in the accumulated data. This t-digest datastructure can also be used in distributed file system. 
If the MVTS dataset is distributed still we can get these parameters using t-digest data structure.

This module performs Exploratory Data Analysis(EDA) on overall MVTS Dataset:  
    a.Missing Value count  
    c.Five-Number summary of each Timeseries  
Creates a summary report and saves in .CSV file in folder specified by user(Example: 'pet_datasets/mvts_analysis/').

Below code snippet shows how to execute different methods of this class,

```python
import CONSTANTS as CONST
import os
from data_analysis.mvts_data_analysis import MVTSDataAnalysis

path_to_root = os.path.join('..', CONST.IN_PATH_TO_MVTS)
mvts = MVTSDataAnalysis(path_to_root)

mvts.compute_summary(CONST.CANDIDATE_PHYS_PARAMETERS)
mvts.print_summary('mvts_eda.csv')
five_num_sum = mvts.get_six_num_summary()
null_count = mvts.get_missing_value()
print('Print Five point Summary')
print(five_num_sum)
print('Print Missing Value on Test Data:')
print(null_count)
mvts.get_six_num_summary() 
```

#### Extracted Features Analysis[[utils.extracted_features_analysis.py](./data_analysis/extracted_features_analysis.py)]
This analyses the dataset(extracted_features.csv) created by feature extractor module.

Perform Exploratory Data Analysis(EDA) on extracted features  
    a.Histogram of classes  
    b.Missing Value count  
    c.Five-Number summary of each Timeseries  
Creates a summary report and saves in .CSV file in folder specified by user(Example: 'pet_datasets/mvts_analysis/').

Below code snippet shows how to execute different methods of this class,

```python
import CONSTANTS as CONST
import pandas as pd
import os
from data_analysis.extracted_features_analysis import ExtractedFeaturesAnalysis

path_to_data = CONST.OUT_PATH_TO_EXTRACTED_FEATURES
path_to_data = os.path.join(path_to_data, "extracted_features.csv")
df = pd.read_csv(path_to_data, sep='\t')

# Testing ExtractedFeaturesAnalysis class on original extracted_features.csv
m = ExtractedFeaturesAnalysis(df, "pet_datasets/mvts_analysis/")
# compute_summary() needs to be called first in order to print/save any Data Analysis results
m.compute_summary()
m.print_summary('extracted_feature_eda.csv')
class_population = m.get_class_population
miss_pop = m.get_missing_values()
five_num = m.get_six_num_summary()
print('Print Class Population:')
print(class_population)
print('Print Missing Value:')
print(miss_pop)
print('Print Five Num Value:')
print(five_num)
```
#### Meta Data Getter[[utils.meta_data_getter.py](./utils/meta_data_getter.py)]
This section is responsible for extracting the embedded information from the raw MVTS file names. These helper methods are used
to fill the first four columns of the extracted_features.csv file: `ID`,`LABEL`,`START_TIME`, and `END_TIME`

#### MVTS Cleaner[[utils.mvts_cleaner.py](./utils/mvts_cleaner.py)]
Responsible for cleaning the dataset by different methods like: interpolation.

#### Normalizer [[utils.normalizer.py](normalizing/normalizer.py)]
Uses different normalization techniques like: zero one normalization, -1 to 1 normalization, standardization and robust standardization. User can choose from these given options and implement on MVTS dataset.

----
#### Authors:

|                 |               |       |
| --------------- |:-------------:| -----:|
| Kankana Sinha   | _ksinha3[AT]student[DOT]gsu[DOT]edu_  | [LinkedIn](https://www.linkedin.com/in/kankana-sinha-4b4b13131/) |
| Azim Ahmadzadeh | _aahmadzadeh1[AT]cs[DOT]gsu[DOT]edu_  | [Website](https://grid.cs.gsu.edu/~aahmadzadeh1)   |