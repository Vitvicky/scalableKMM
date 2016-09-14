# scalableKMM
The implement of the KMM algorithm's scalable/spark version.

The project is to implement a scalable version of a solution to a problem known as sampling bias correction. In machine learning, a training set is used to learn a hypothesis. This hypothesis is tested on a test dataset assuming that the given dataset exhibits similar data distribution as the training set. However in many real-world scenarios, this assumption does not hold. Empirical data distribution of the training and test set may be different.

Generally, if the two datasets are unrelated, it is not possible to learn a good model (hypothesis). But if you consider that the class conditional probability distributions are the same between training and test datasets, i.e. P(Y | X) are the same, then the only change is the marginal distribution P(X).

1.X is the set of features and Y is the class label. Note that P(X,Y) = P(Y|X) * P(X).
In this scenario, it is possible to consider P(X_te) = B * P(X_tr).

2.X_tr = data instances (covariates) from training data.

3.X_te = data instances (covariates) from test data.

4.B = constant.

The problem is to find B.

Multiple methods have been proposed to solve for B. Particularly, a method known as "Kernel Mean Matching" approximates B with a quadratic program. Unfortunately, the quadratic program does not scale well with large datasets, especially when training set size is large. Therefore, we can use a common technique called bagging (bootstrap with aggregation) to address this challenge. Bagging is a technique in machine learning that is known to perform well on unstable estimators. In this case, the basic idea is to sample a subset of data instances and estimate required values. Repeat this for a sufficiently large number of samples, and combine the estimations from each sample to give a final output.

Considering that bagging is applicable on both training and test datasets, your primary task is to design a strategy so that this quadratic program (which is the estimator in question) can be run on large number of samples from both training and test datasets on Apache Spark. We will provide you with the datasets and a basic quadratic program (in python) to perform Kernel Mean Matching. You are to design the bagging mechanism and data synchronization to compute the overall result.

For more details on the problem, please refer to the following papers.
1) Gretton, Arthur, Alex Smola, Jiayuan Huang, Marcel Schmittfull, Karsten Borgwardt, and Bernhard Schölkopf. "Covariate shift by kernel mean matching." Dataset shift in machine learning 3, no. 4 (2009): 5.
2) Miao, Yun-Qian, Ahmed K. Farahat, and Mohamed S. Kamel. "Ensemble Kernel Mean Matching." In Data Mining (ICDM), 2015 IEEE International Conference on, pp. 330-338. IEEE, 2015.
