# EqualizationEnsemble


**Equalization ensemble method (EASE) is a general learning framework for large scale highly imbalanced Data.**


**This repository contains:**
- Implementation of Equalization Ensemble
- Implementation of Self-paced Ensemble [1]
- Implementation of Entropy and Confidence-based Under-sampling Boosting [2]
- Implementation of Hashing-based Under-sampling Ensemble [3]
- Implementation of Radial-Based Undersampling [4]
- Implementation of 5 ensemble-based imbalance learning baselines
  - `SMOTEBoost` [5]
  - `SMOTEBagging` [6]
  - `RUSBoost` [7]
  - `UnderBagging` [8]
  - `BalanceCascade` [9]
- Implementation of 15 resampling based imbalance learning baselines
- Additional experimental results

**NOTE:** The implementations of [5],[7], Polynom fit SMOTE [10] and resampling methods are based on [imbalanced-algorithms](https://github.com/dialnd/imbalanced-algorithms), [smote_variants] https://smote-variants.readthedocs.io/en/latest/installation.html and [imbalanced-learn](https://github.com/scikit-learn-contrib/imbalanced-learn).



# Background

we have proposed the framework of EASE to address the imbalanced classification problems. In the framework, the binning-based equalization under-sampling method has been used to provide balanced data sets for each of the base classifiers and combines the weighted integration strategy by using G-mean score as weights to improve the diversity and performance of the base classifiers at the same time. The extensive experiments have shown that the performance of the proposed method is not only significantly better than the contending methods on 61 small-scale data sets with low IR(<130) (especially for F1 and MMC metric), but also superior to the methods using the under-sampling technique on larger-scale data sets with high IR(>270). The figure below gives an overview of the EASE framework.

![image](https://github.com/jinjunren/ease/figure/framework.png)

# Install

Our EASE implementation requires following dependencies:
- [python](https://www.python.org/) (>=3.7)
- [numpy](https://numpy.org/) (>=1.11)
- [scipy](https://www.scipy.org/) (>=0.17)
- [scikit-learn](https://scikit-learn.org/stable/) (>=0.21)
- [imblearn](https://pypi.org/project/imblearn/) (>=0.2) (optional, for canonical resampling)


```
git clone https://
```

# Usage

## Documentation

**Our EASE implementation can be used much in the same way as the ensemble classifiers in [sklearn.ensemble](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble).**  

| Parameters    | Description   |
| ------------- | ------------- |
| `base_estimator` | *object, optional (default=`sklearn.tree.DecisionTreeClassifier()`)* <br> The base estimator to fit on self-paced under-sampled subsets of the dataset. NO need to support sample weighting. Built-in `fit()`, `predict()`, `predict_proba()` methods are required. |
| `maj_cls_prob`  | *function, optional (default=`lambda y_true, y_pred: np.absolute(y_true-y_pred)`)* <br> User-specified classification hardness function. <br> Input: `y_true` and `y_pred` Output: `hardness` (1-d array)  |
| `n_estimator`    | *integer, optional (default=10)* <br> The number of base estimators in the ensemble. |

| `random_state`   | *integer / RandomState instance / None, optional (default=None)* <br> If integer, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by `numpy.random`. |

----------------

| Methods    | Description   |
| ---------- | ------------- |
| `fit(self, X, y, label_maj=0, label_min=1)` | Build a self-paced ensemble of estimators from the training set (X, y). <br> `label_maj`/`label_min` specify the label of majority/minority class. <br> By default, we let the minority class be positive class (`label_min=1`). |
| `predict(self, X)` | Predict class for X. |
| `predict_proba(self, X)` | Predict class probabilities for X. |
| `predict_log_proba(self, X)` | Predict class log-probabilities for X. |
| `score(self, X, y)` | Returns the average precision score on the given test data and labels. |

----------------

| Attributes    | Description   |
| ------------- | ------------- |
| `base_estimator_` | *estimator* <br> The base estimator from which the ensemble is grown. |
| `estimators_` | *list of estimator* <br> The collection of fitted base estimators. |
