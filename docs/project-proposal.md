Module 5: Homework 0
Ryan Christ and Owen Sizemore

J. C. Garnett, “Prediction of Mohs Hardness with Machine Learning Methods Using Compositional Features,” ACS symposium series, pp. 23–48, Jan. 2019, doi: https://doi.org/10.1021/bk-2019-1326.ch002.

This paper uses a dataset of 622 naturally-occurring minerals and 51 man-made artificial minerals and predicts Mohs hardness using 11 features, applying nine different machine learning models and comparing results. The features are described below in Table 2 from the paper.



The dataset is publicly available at https://data.mendeley.com/datasets/jm79zfps6b/1. In our replication study, we plan to replicate Figures 3 and 4 from the paper, where Figure 3 compares performance across models used and Figure 4 are ROC plots comparing the effectiveness of binary classifiers to OVR classifiers. The paper develops its nine ML models through a variety of techniques, including binary and ternary classification with RBF kernels, random forests, and SVMs. We will similarly need to build and train nine models, replicating the process of using the 622 naturally-occurring minerals as training data and the 51 artificial minerals as test data.



For an extension to the paper described above, we propose incorporating an artificial neural network to improve prediction of Mohr's hardness from compositional features. The original study mainly uses random forest (RF) models and does not extensively investigate how neural network architecture affects the performance of the model. In our extension we plan to vary the architecture to find the optimal neural network, while using different regularization techniques, to compare the performance of this model against the baseline methods shown in the paper. This will help us understand which machine learning models can best capture relationships between composition and material hardness. 
‌
