from flight_ad.datasets import load_dashlink_bindings
from flight_ad.utils.data import DataBinder
from flight_ad.wrangling import DataWrangler
from wrangling_functions import preprocess, change_col, resample, select
from flight_ad.transformations import reshape_df_interspersed
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from flight_ad.cluster import DBSCAN
from flight_ad.learn import FunctionEstimator
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
