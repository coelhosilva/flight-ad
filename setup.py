import setuptools

VERSION = "0.0.1"
DESCRIPTION = "flight-ad is a Python package for anomaly detection in the aviation domain built on top of scikit-learn."
with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="flight_ad",
    version=VERSION,
    author="Lucas Coelho e Silva",
    author_email="lucascoelhosilva@gmail.com",
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/coelhosilva/flight-ad.git",
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
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    include_package_data=True,
)
