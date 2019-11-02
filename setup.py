from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

setup(
    name="TimeSeriesAnalyzer",
    version="0.0.1",
    author="Kankana Sinha & Azim Ahmadzadeh",
    author_email="ksinha1106@gmail.com",
    description="A package to analyse time series data",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://bitbucket.org/azimdmlab/mvts_features/src/master/",
    packages=find_packages(),
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
    ],
)