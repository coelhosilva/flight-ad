import setuptools
import os

NAME = "flight-ad"
VERSION = "0.0.1"
DESCRIPTION = "flight-ad is a Python package for anomaly detection in the aviation domain built on top of scikit-learn."
package_root = os.path.abspath(os.path.dirname(__file__))
readme_filename = os.path.join(package_root, "README.md")

with open(readme_filename, "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name=NAME,
    version=VERSION,
    author="Lucas Coelho e Silva",
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/coelhosilva/flight-ad.git",
    download_url="https://github.com/coelhosilva/flight-ad/archive/refs/tags/v0.0.1.tar.gz",
    license='MIT',
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy~=1.20.3',
        'scikit-learn~=0.24.2',
        'pandas~=1.2.4',
        'matplotlib~=3.4.2',
        'pyarrow~=4.0.1',
        'tqdm~=4.61.1'
    ],
    keywordsList=['anomaly detection', 'anomaly', 'flight'],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.8',
    include_package_data=True,
)
