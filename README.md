## MVTS Data Toolkit
Multivariate Time Series Data Toolkit is a python package that works on multivariate time series datasets and provides: 

 - over 50 time series statistical features collected from a number of research studies of different domains,
 - an automated feature extraction process, provided in both sequential and parallel fashions,
 - a set of generic sampling methodologies,
 - a set of different normalization transformations on mvts data,
 - an automated data analysis process that provides basic summary on both mvts data and the extracted features.

----
[![PyPI license](https://img.shields.io/pypi/l/ansicolortags.svg??style=flat-square&logo=appveyor)](https://pypi.org/project/TimeSeriesAnalyzer/)
[![PyPI license](https://img.shields.io/badge/PyPI-0.0.1-orange??style=flat-square&logo=appveyor)](https://pypi.org/project/TimeSeriesAnalyzer/)
[![PyPI license](https://img.shields.io/badge/Doc-Sphinx-blue??style=flat-square&logo=appveyor)](http://dmlab.cs.gsu.edu/docs/mvtsdata_toolkit/)
----
 
#### Requirements
* Python > 3.6
* For a list of all required packages, see [requirements.txt](./requirements.txt).

----
#### Try it online:
Click on the badge below to try the demo provided in the notebook `demo.ipynb`:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/git/https%3A%2F%2Fbitbucket.org%2Fgsudmlab%2Fmvtsdata_toolkit/87636b0eca70ba0ebac2629ad02f5c08932c86f2)

----
#### Install it from PyPI
You can install this package, directly from Python Package Index (PyPI), using `pip` as follows:

```pip install mvts_data_toolkit```

[![PyPI license](https://img.shields.io/badge/PyPI-0.0.1-orange??style=flat-square&logo=appveyor)](https://pypi.org/project/TimeSeriesAnalyzer/)


----
#### See Documentation
Check out the documentation of the project here:

[![PyPI license](https://img.shields.io/badge/Doc-Sphinx-blue??style=flat-square&logo=appveyor)](http://dmlab.cs.gsu.edu/docs/mvtsdata_toolkit/)

----

### Multivariate Timeseries (MVTS) Data
## Data Rules:
#### MVTS Files
It is assumed that the input data follows these assumptions:

1. Each mvts is stored in a `tab` delimited, csv file. Each
column represents a time series, except the first column which
has the time stamp of each value. (A mvts data with `t` time
series, each of length `d`, has a dimension of `d X (t+1)`)

2. File names should have the following naming convention,  
   i. LABEL: located between `'lab['` and the first occurrence of `']'` after that.  
   ii. ID: located between `'id['` and the first occurrence of `']'` after that.  
   iii. START TIME: located between `'st['` and the first occurrence of `']'` after that.  
   iv. END TIME: located between `'et['` and the first occurrence of `']'` after that.
   
   So, an example would be:  
   ```lab[B]1.0@1053_id[345]_st[2011-01-24T03:24:00]_et[2011-01-24T11:12:00].csv```
  
3. If the embedded values (`label`, `id`, ...) contain paired braces `'[]'` within the string, 
(e.g. for start date, `st[2011-01-24T[03:24:00]]`, then the methods will be able to extract
 correctly but for unpaired brace it will raise an exception.
 
4. It is optional to embed the id of each mvts in their filename. 
The order of the embedding doesn't matter as long as the starting expression and braces'[]' are placed correctly. 

#### Extracted Features Files
It is assumed that the features extracted from all mvts data follow
these assumptions:

1. The extracted features will be stored in a `tab` delimited,
csv file.
2. Each row summarizes one mvts with a list of features.
3. The first four columns of this file are reserved for: `ID`,
`LABEL`,`START_TIME`, and `END_TIME`. The dimension of the
extracted features dataframe will be `n X f` where `n` is the
number of mvts, and `f` is the number of chosen features.

----

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