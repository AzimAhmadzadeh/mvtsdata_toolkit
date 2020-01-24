from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

setup(
    name="mvtsdata_toolkit",
    version="0.1.3",
    author="Azim Ahmadzadeh, Kankana Sinha",
    author_email="aahmadzadeh1@cs.gsu.edu, ksinha1106@gmail.com",
    maintainer="Azim Ahmadzadeh",
    maintainer_email="aahmadzadeh1@cs.gsu.edu",
    description="A Toolkit for Multivariate Time Series Data",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://bitbucket.org/gsudmlab/mvtsdata_toolkit/src/master/",
    packages=find_packages(),
    py_modules=['CONSTANTS'],
    include_package_data=True,
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
)
