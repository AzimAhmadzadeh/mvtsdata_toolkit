import os
from setuptools import setup, find_packages
import CONSTANTS as CONST

# ------------ VARIABLES ------------
readme_fname = 'README.md'
requirement_fname = 'requirements.txt'

readme_path = os.path.join(CONST.ROOT, readme_fname)
requirement_path = os.path.join(CONST.ROOT, requirement_fname)

# ------------ SCRIPTS --------------
with open(readme_path, "r") as readme_file:
    readme = readme_file.read()

if os.path.isfile(requirement_path):
    with open(requirement_path) as f:
        install_requires = f.read().splitlines()

pckges = find_packages(exclude=['tests.*', 'tests', 'docs.*', 'docs'])
# pckges = find_packages()

# ------------- SETUP ---------------
setup(
    name='mvtsdatatoolkit',
    version='0.2.6',
    author='Azim Ahmadzadeh, Kankana Sinha',
    author_email='aahmadzadeh1@cs.gsu.edu',  #  ksinha1106@gmail.com
    url='https://bitbucket.org/gsudmlab/mvtsdata_toolkit/src/master',
    maintainer='Azim Ahmadzadeh',
    maintainer_email='aahmadzadeh1@cs.gsu.edu',
    description='A Toolkit for Multivariate Time Series Data',
    long_description=readme,
    long_description_content_type='text/markdown',
    packages=pckges,
    package_data={
        '.': ['requirements.txt',
              'requirements_with_transitive.txt'],
        'mvtsdatatoolkit.configs': ['datasets_configs.yml',
                                    'demo_configs.yml',
                                    'test_configs.yml']},
    install_requires=install_requires,
    py_modules=['CONSTANTS'],
    include_package_data=True,
    license='MIT',
    keywords=['multivariate', 'time series', 'mvts', 'imbalance', 'sampling', 'features'],
    project_urls={
        'Documentation': 'http://dmlab.cs.gsu.edu/docs/mvtsdata_toolkit/',
        'Source': 'https://bitbucket.org/gsudmlab/mvtsdata_toolkit/src/master',
    },
    classifiers=[
        # Keys & values must be chosen from: https://pypi.org/classifiers/
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Development Status :: 4 - Beta',  # 3-Alpha, 4-Beta, 5-Production/Stable
        'License :: OSI Approved :: MIT License',
    ],
)
