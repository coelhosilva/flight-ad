---
title: 'flight-ad: Flight analytics framework for anomaly detection in Python.'
tags:
  - flight data analysis
  - anomaly detection
  - outlier detection
  - machine learning
  - unsupervised learning

authors:
 - name: Lucas Coelho e Silva
   orcid: 0000-0002-6499-912X
   affiliation: 1
 - name: Mayara Condé Rocha Murça
   orcid: N/A
   affiliation: 1
affiliations:
 - name: Civil Engineering Department (IEI), Aeronautics Institute of Technology (ITA)
   index: 1
date: 1 July 2021
bibliography: paper.bib
---

# Summary

One of the next challenges regarding flight safety is the change from an incident-based reactive approach to one which integrates a more proactive system-based predictive approach, that aims at identifying a hazard before its consequences happen, either fully or partially.

In parallel, aviation systems generate more and more data. From the ATC standpoint, modern and more widespread surveillance equipment provides detailed flight information. From the aircraft operator perspective, better-equipped airplanes with modern flight data recorders (e.g., QAR and DFDR) are capable of registering more parameters and at higher sampling rates.

Within this context of abundant data and focus on increasing safety levels via more proactive approaches while enhancing operational efficiency, data mining initiatives for understanding and predicting aviation system behavior have become more prominent. One of these initiatives, which has drawn attention over the past years, is anomaly detection.

There are various methodologies for anomaly detection within the flight data context. Over the past years, several studies on this matter have been performed. Nevertheless, they all discuss study-specific steps for solving the anomaly detection problem. Therefore, an extensive framework for the application of anomaly detection techniques in flight operations data is lacking.

``flight-ad`` is a Python 3 package for anomaly detection in the 
aviation domain built on top of scikit-learn [@scikit-learn].

It provides:

-  An implementation of an anomaly detection pipeline;
-  A DataBinder object for loading and transforming the data within the pipeline on the fly;
-  A DataWrangler object for building a data wrangling pipeline;
-  A StatisticalLearner object for binding scikit-learn's pipelines and integrating them on the anomaly detection workflow;
-  Visualization tools for assessing potential anomalies;
-  Reporting tools for analyzing results;
-  Sample airplane sensor data (repackaged from [@dashlink] "Sampled Fight Data" dataset for the purpose of evaluating and advancing data mining capabilities that can be used to promote aviation safety;
-  Adaptations of machine learning algorithms, such as a DBSCAN implementation that calculates the hyperparameter epsilon from the input data.


# Research

``flight-ad`` is currently being used in the following research:

  -  Should we add a pre-print version to, e.g., arXiv so we have a reference and don't have to wait for a full cycle of the paper?
  
# References