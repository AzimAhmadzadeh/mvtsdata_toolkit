from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

setup(
    name="MVTS Data Toolkit",
    version="0.0.1",
    author="Kankana Sinha - Azim Ahmadzadeh",
    author_email="ksinha3@student.gsu.edu - aahmadzadeh1@cs.gsu.edu",
    description="A python software to facilitate working with multivariate time series datasets.",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://AzimAhmadzadeh@bitbucket.org/gsudmlab/mvts_data_toolkit.git",
    packages=find_packages(),
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
)