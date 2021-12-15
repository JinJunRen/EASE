# -*- coding: utf-8 -*-
"""
Created on Tue May 14 14:32:27 2019
@author: ZhiningLiu1998
mailto: znliu19@mails.jlu.edu.cn / zhining.liu@outlook.com
"""

import numpy as np
import scipy.sparse as sp
import sklearn
import logging
from sklearn.tree import DecisionTreeClassifier
#from tools.imbalancedmetrics import ImBinaryMetric
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings
warnings.filterwarnings("ignore")

class SelfPacedEnsemble(BaseEstimator, ClassifierMixin):
    """ Self-paced Ensemble (SPE)

    Parameters
    ----------

    base_estimator : object, optional (default=sklearn.Tree.DecisionTreeClassifier())
        The base estimator to fit on self-paced under-sampled subsets of the dataset. 
        NO need to support sample weighting. 
        Built-in `fit()`, `predict()`, `predict_proba()` methods are required.

    hardness_func :  function, optional 
        (default=`lambda y_true, y_pred: np.absolute(y_true-y_pred)`)
        User-specified classification hardness function
            Parameters:
                y_true: 1-d array-like, shape = [n_samples] 
                y_pred: 1-d array-like, shape = [n_samples] 
            Returns:
                hardness: 1-d array-like, shape = [n_samples]

    n_estimators :  integer, optional (default=10)
        The number of base estimators in the ensemble.

    k_bins :        integer, optional (default=10)
        The number of hardness bins that were used to approximate hardness distribution.

    random_state :  integer / RandomState instance / None, optional (default=None)
        If integer, random_state is the seed used by the random number generator; 
        If RandomState instance, random_state is the random number generator; 
        If None, the random number generator is the RandomState instance used by 
        `numpy.random`.


    Attributes
    ----------

    base_estimator_ : estimator
        The base estimator from which the ensemble is grown.

    estimators_ : list of estimator
        The collection of fitted base estimators.

    Example:
    ```
    import numpy as np
    from sklearn import datasets
    from sklearn.tree import DecisionTreeClassifier
    from self_paced_ensemble import SelfPacedEnsemble
    from utils import make_binary_classification_target, imbalance_train_test_split

    X, y = datasets.fetch_covtype(return_X_y=True)
    y = make_binary_classification_target(y, 7, True)
    X_train, X_test, y_train, y_test = imbalance_train_test_split(
            X, y, test_size=0.2, random_state=42)

    def absolute_error(y_true, y_pred):
        # Self-defined classification hardness function
        return np.absolute(y_true - y_pred)

    spe = SelfPacedEnsemble(
        base_estimator=DecisionTreeClassifier(),
        hardness_func=absolute_error,
        n_estimators=10,
        k_bins=10,
        random_state=42,
    ).fit(
        X=X_train,
        y=y_train,
    )
    print('auc_prc_score: {}'.format(spe.score(X_test, y_test)))
    ```

    """
    base_estimator_=DecisionTreeClassifier()
    def __init__(self, 
            base_estimator=DecisionTreeClassifier(), 
            hardness_func=lambda y_true, y_pred: np.absolute(y_true-y_pred),
            n_estimators=10,
            showlogger=False,
            k_bins=10, 
            random_state=None):
        self.base_estimator_ = base_estimator
        self.estimators_ = []
        self.hardness_func = hardness_func
        self.n_estimators = n_estimators
        self.k_bins = k_bins
        self.showlogger = showlogger
        self.random_state = random_state
        self.logger = logging.getLogger("mainModule.M_SelfPacedEnsemble")
    @classmethod
    def fit_base_estimator(self, X, y):
        """Private function used to train a single base estimator.需要用属性获得分类器的名，用属性获得有问题，有可能属性对象没值"""
        return sklearn.base.clone(self.base_estimator_).fit(X, y)

    def random_under_sampling(self, X_maj, y_maj, X_min, y_min):
        """Private function used to perform random under-sampling."""
        np.random.seed(self.random_state)
        idx = np.random.choice(len(X_maj), len(X_min), replace=False)
        X_train = np.concatenate([X_maj[idx], X_min])
        y_train = np.concatenate([y_maj[idx], y_min])
        return X_train, y_train

    def self_paced_under_sampling(self, 
            X_maj, y_maj, X_min, y_min, i_estimator):
        """Private function used to perform self-paced under-sampling."""
        # Update hardness value estimation
        hardness = self.hardness_func(y_maj, self._y_pred_maj)

        # If hardness values are not distinguishable, perform random smapling
        if hardness.max() == hardness.min():
            X_train, y_train = self.random_under_sampling(X_maj, y_maj, X_min, y_min)
        # Else allocate majority samples into k hardness bins
        else:
            step = (hardness.max()-hardness.min()) / self.k_bins
            bins = []; ave_contributions = []
            for i_bins in range(self.k_bins):
                idx = (
                    (hardness >= i_bins*step + hardness.min()) & 
                    (hardness < (i_bins+1)*step + hardness.min())
                )
                # Marginal samples with highest hardness value -> kth bin
                if i_bins == (self.k_bins-1):
                    idx = idx | (hardness==hardness.max())
                bins.append(X_maj[idx])
                ave_contributions.append(hardness[idx].mean())

            # Update self-paced factor alpha
            alpha = np.tan(np.pi*0.5*(i_estimator/(self.n_estimators-1)))
            # Caculate sampling weight
            weights = 1 / (ave_contributions + alpha)
            weights[np.isnan(weights)] = 0
            # Caculate sample number from each bin
            n_sample_bins = len(X_min) * weights / weights.sum()
            n_sample_bins = n_sample_bins.astype(int)+1
            nl=list()
            n=0
            for b in bins:
                t=len(b)
                n+=t
                nl.append(t)
            if self.showlogger:
                self.logger.warning("\nave_contributions:%s"%(str(ave_contributions)))
                self.logger.warning("hardness:%s"%(str(hardness)))
                self.logger.warning("iteroator:%s,欲采样本数:%s"%(i_estimator,str(n_sample_bins)))
                self.logger.warning("实际拥有样本数:%s"%(str(nl)))
            # Perform self-paced under-sampling
            sampled_bins = []
            for i_bins in range(self.k_bins):
                if min(len(bins[i_bins]), n_sample_bins[i_bins]) > 0:
                    np.random.seed(self.random_state)
                    idx = np.random.choice(
                        len(bins[i_bins]), 
                        min(len(bins[i_bins]), n_sample_bins[i_bins]), 
                        replace=False)
                    sampled_bins.append(bins[i_bins][idx])
            X_train_maj = np.concatenate(sampled_bins, axis=0)
            y_train_maj = np.full(X_train_maj.shape[0], y_maj[0])
            # Handle sparse matrix
            if sp.issparse(X_min):
                X_train = sp.vstack([sp.csr_matrix(X_train_maj), X_min])
            else:
                X_train = np.vstack([X_train_maj, X_min])
            y_train = np.hstack([y_train_maj, y_min])
            
        return X_train, y_train

    def fit(self, X, y, label_maj=0, label_min=1):
        """Build a self-paced ensemble of estimators from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        y : array-like, shape = [n_samples]
            The target values (class labels).
        
        label_maj : int, bool or float, optional (default=0)
            The majority class label, default to be negative class.
            
        label_min : int, bool or float, optional (default=1)
            The minority class label, default to be positive class.
        
        Returns
        ------
        self : object
        """
        self.estimators_ = []
        # Initialize by spliting majority / minority set
        X_maj = X[y==label_maj]; y_maj = y[y==label_maj]
        X_min = X[y==label_min]; y_min = y[y==label_min]

        # Random under-sampling in the 1st round (cold start)
        X_train, y_train = self.random_under_sampling(
            X_maj, y_maj, X_min, y_min)
        self.estimators_.append(
            self.fit_base_estimator(
                X_train, y_train))
        self._y_pred_maj = self.predict_proba(X_maj)[:, 1]
        # Loop start
        for i_estimator in range(1, self.n_estimators):
            X_train, y_train = self.self_paced_under_sampling(
                X_maj, y_maj, X_min, y_min, i_estimator,)
            self.estimators_.append(
                self.fit_base_estimator(
                    X_train, y_train))
            # update predicted probability
            n_clf = len(self.estimators_)
            y_pred_maj_last_clf = self.estimators_[-1].predict_proba(X_maj)[:, 1]
            self._y_pred_maj = (self._y_pred_maj * (n_clf-1) + y_pred_maj_last_clf) / n_clf

        return self

    def predict_proba(self, X):
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample is computed as
        the mean predicted class probabilities of the base estimators in the
        ensemble. If base estimators do not implement a ``predict_proba``
        method, then it resorts to voting and the predicted class probabilities
        of an input sample represents the proportion of estimators predicting
        each class.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. 
        """
        y_pred = np.array(
            [model.predict(X) for model in self.estimators_]
            ).mean(axis=0)
        if y_pred.ndim == 1:
            y_pred = y_pred[:, np.newaxis]
        if y_pred.shape[1] == 1:
            y_pred = np.append(1-y_pred, y_pred, axis=1)
        return y_pred
    
    def predict(self, X):
        """Predict class for X.

        The predicted class of an input sample is computed as the class with
        the highest mean predicted probability. If base estimators do not
        implement a ``predict_proba`` method, then it resorts to voting.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes.
        """
        y_pred_binarized = sklearn.preprocessing.binarize(
            self.predict_proba(X)[:,1].reshape(1,-1), threshold=0.5)[0]
        return y_pred_binarized
    
    def score(self, X, y):
        """Returns the average precision score (equivalent to the area under 
        the precision-recall curve) on the given test data and labels.
        
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.

        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True labels for X.

        Returns
        -------
        score : float
            Average precision of self.predict_proba(X)[:, 1] wrt. y.
        """
        return sklearn.metrics.average_precision_score(
            y, self.predict_proba(X)[:, 1])