# flight-ad

[![Codacy Badge](https://app.codacy.com/project/badge/Grade/d2f06dedcb044256828e1c907d9c511a)](https://www.codacy.com/gh/coelhosilva/flight-ad/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=coelhosilva/flight-ad&amp;utm_campaign=Badge_Grade)

`flight-ad` is a Python package for anomaly detection in the aviation domain built on top of scikit-learn.

It provides:

-   An implementation of an anomaly detection pipeline;
-   A DataBinder object for loading and transforming the data within the pipeline on the fly;
-   A DataWrangler object for building a data wrangling pipeline;
-   A StatisticalLearner object for binding scikit-learn's pipelines and integrating them on the anomaly detection workflow;
-   Visualization tools for assessing potential anomalies;
-   Reporting tools for analyzing results;
-   Sample airplane sensor data, repackaged from NASA's DASHlink for the purpose of evaluating and advancing data mining capabilities that can be used to promote aviation safety;
-   Adaptations of machine learning algorithms, such as a DBSCAN implementation that calculates the hyperparameter epsilon from the input data.

## Installation

The easiest way to install `flight-ad` is using pip from your virtual environment.

Directly from GitHub:

`pip install git+https://github.com/coelhosilva/flight-ad.git`

## Examples

This is a sample usage of the package for constructing an anomaly detection pipeline. Beware that the sample dataset 
may take up roughly 1 GB in disk space.

```python
from flight_ad.datasets import load_dashlink_bindings
from flight_ad.utils.data import DataBinder
from flight_ad.wrangling import DataWrangler
from wrangling_functions import preprocess, change_col, resample, select
from flight_ad.transformations import reshape_df_interspersed
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from flight_ad.cluster import DBSCAN
from flight_ad.learn import FunctionTransformer
from flight_ad.learn import StatisticalLearner
from flight_ad.pipeline import AnomalyDetectionPipeline
from flight_ad.report import clustering_info, silhouette

# Binder
data_bindings = load_dashlink_bindings(download=True)
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
ad_pipeline.fit()

# Results
labels, n_clusters, n_noise = clustering_info(learner.pipeline['dbscan'])
avg_silhouette, _ = silhouette(learner.partial_data['pca'], labels)
```

## Package structure

TBD.

## Dependencies

`flight-ad` requires:

-   Python (>=3.6)
-   NumPy
-   pandas
-   scikit-learn
-   matplotlib
-   tqdm

## Contributions

We welcome and encourage new contributors to help test `flight-ad` and add new functionality. Any input, feedback, 
bug report or contribution is welcome.

If one wishes to contact the author, they may do so by emailing <coelho@ita.br>.

## Citation

If you use `flight-ad` in a scientific publication, we would appreciate citations.

BibTex: TBD.

Citation string: TBD.
