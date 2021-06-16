# flight-ad
**flight-ad** is a Python package for anomaly detection in the aviation domain built on top of scikit-learn.

It provides:

- An implementation of an anomaly detection pipeline;
- Visualization tools for assessing potential anomalies;
- Reporting tools for analyzing model results;
- Sample airplane sensor data (from NASA's DASHlink);
- Sample pipelines such as the DBSCAN pipeline.

## Instalation
The easiest way to install flight-ad is using pip from your virtual environment.

Directly from GitHub:

```pip install git+https://github.com/Flight-Anomaly-Detection/flight-ad.git```

## Examples
TODO.
```python
from flight_ad.unsupervised import DBSCANPipeline, clustering_info
dbscan_pipeline = DBSCANPipeline(flight_vector)
dbscan_pipeline.fit()
results = clustering_info(dbscan_pipeline.pipeline['dbscan'])
```

## Package structure
TODO.

## Dependencies
`flight-ad` requires:
- Python (>=3.6)
- NumPy
- pandas
- scikit-learn
- matplotlib

## Contributions
We welcome and encourage new contributors to help test flight-ad and add new functionality. If one wishes to contact the author, they may do so by emailing coelho@ita.br.


## Citation
If you use flight-ad in a scientific publication, we would appreciate citations.
BibTex: add.
Citation string: add.

<!-- ```pip install flight-ad ``` -->
<!-- ---------------------- -->
<!-- <hr style="border:2px solid gray"> </hr> -->