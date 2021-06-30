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
from flight_ad.datasets import load_dashlink_bindings
from flight_ad.utils.data import DataBinder
from flight_ad.wrangling import DataWrangler
from wrangling import preprocess, change_col, resample, select
from flight_ad.transformations import reshape_df_interspersed
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from flight_ad.unsupervised import DBSCAN
from flight_ad.learn import FunctionEstimator
from flight_ad.learn import StatisticalLearner
from flight_ad.pipeline import AnomalyDetectionPipeline
from flight_ad.report import clustering_info, silhouette

# Binder
data_bindings = load_dashlink_bindings()
binder = DataBinder(data_bindings)

# Wrangler
wrangling_steps = [
    ('preprocess_flight', preprocess),
    ('resample_dataframe', resample),
    ('change_col', change_col),
    ('select_col', select)

]
wrangler = DataWrangler(wrangling_steps, memorize='change_col')

# Learner
learning_steps = {
    'preprocessing': [
        ('reshaper', FunctionEstimator(reshape_df_interspersed)),
        ('scaler', StandardScaler()),
        ('pca', PCA())
    ],
    'training': [
        ('dbscan', DBSCAN())
    ]
}
learner = StatisticalLearner(learning_steps, record='pca')

# Pipeline
ad_pipeline = AnomalyDetectionPipeline(binder, wrangler, learner)
ad_pipeline.run()

# Results
labels, n_clusters, n_noise = clustering_info(learner.pipeline['dbscan'])
avg_silhouette, _ = silhouette(learner.partial_data['pca'], labels)
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