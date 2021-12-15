import numpy as np
import sys
from sklearn.base import BaseEstimator, ClassifierMixin
from imblearn.under_sampling import RandomUnderSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import binarize


class ECUBoostRF(BaseEstimator, ClassifierMixin):

    def __init__(self, L=100, lamda=0.05, k=5,T=50):
        """
        Training data X with Nn majority and Np minority,
        ensemble iterative number L,
        weight coefficient λ between the confidence and entropy,
        nearest neighbor parameter k for entropy calculation
        tree number T of RF.
        """
        self.L = L
        self.lamda = lamda
        self.k = k
        self.T = T

    def fit(self, X, y):
        assert ({0, 1} == set(y)), "the class label is not {0，1}. Note:majority class label-0,minority class lable --1"
        self.min_ind = set(np.argwhere(y == 1).ravel())
        self.maj_ind = set(np.argwhere(y == 0).ravel())
        self.X = X.copy()
        self.y = y.copy()
        self.cal_entropy()
        self.clf = self.train()

    def predict_proba(self, X):
        y_pred = np.array(
            [clf.predict(X) for clf in self.clf]
        ).mean(axis=0)
        if y_pred.ndim == 1:
            y_pred = y_pred[:, np.newaxis]
        if y_pred.shape[1] == 1:
            y_pred = np.append(1 - y_pred, y_pred, axis=1)
        return y_pred

    def predict(self, X):
        y_pred_binarized = binarize(
            self.predict_proba(X)[:, 1].reshape(1, -1), threshold=0.5)[0]
        return y_pred_binarized

    def train(self):
        L = 0
        classifier = []
        # ---------------------------step 1 cal entropy-------------------------
        self.cal_entropy()

        # ---------------------------step 2 get f0 (RandomForest)---------------
        rus_sampled_X, rus_sampled_y = self.RandomUndersampling()
        RF0 = RandomForestClassifier(n_estimators=self.T)
        RF0.fit(rus_sampled_X, rus_sampled_y)
        classifier.append(RF0)
        L = 1

        while L <= self.L:
            # ----------------------step 3 Obtain Conf--------------------------
            maj_conf = [0] * len(self.maj_ind)
            proba_list = [clf.predict_proba(self.X) for clf in classifier]
            for proba in proba_list:
                maj_conf_list_tmp = [proba[idx][0] for idx in self.maj_ind]
                maj_conf = [maj_conf_list_tmp[i] + maj_conf[i] for i in range(0, len(maj_conf))]
            maj_conf_max = max(maj_conf)
            maj_conf_min = min(maj_conf)
            normed_maj_conf = [(val - maj_conf_min) / (maj_conf_max - maj_conf_min) for val in maj_conf]
            idx2conf = {maj_idx: normed_maj_conf[conf_idx] for conf_idx, maj_idx in enumerate(self.maj_ind)}
            idx2rank = {}
            for maj_idx in self.maj_ind:
                idx2rank.update({maj_idx: (1 - self.lamda) * idx2conf[maj_idx] +
                                self.lamda * self.idx2entropy_dict[maj_idx]})
            sorted_idx2rank = sorted(idx2rank.items(), key=lambda x: x[1], reverse=False)

            # ---------step 4 select X_l(same num as N_p) from X_n with with lowest Rank----
            sampled_X_idx = [val[0] for val in sorted_idx2rank[:len(self.min_ind)]]
            exclude_x_idx = [val for val in self.maj_ind if val not in sampled_X_idx]
            sampled_X = np.delete(self.X.copy(), exclude_x_idx, axis=0)
            sampled_y = np.delete(self.y.copy(), exclude_x_idx, axis=0)
            RF = RandomForestClassifier(n_estimators=self.T)
            RF.fit(sampled_X, sampled_y)
            classifier.append(RF)
            L += 1
        return classifier

    def cal_entropy(self):
        KneigborModel = KNeighborsClassifier(n_neighbors=self.k)
        KneigborModel.fit(self.X.copy(), self.y.copy())
        neigh_dist, neigh_ind = KneigborModel.kneighbors(self.X.copy(), self.k + 1, True)
        entropy_max = -np.log(0.5)
        neigh_dist = neigh_dist[:, 1:]
        neigh_ind = neigh_ind[:, 1:]
        entropy_cer_list = []
        entropy_str_list = []
        entropy_cer_max = float('-inf')
        entropy_cer_min = float('inf')
        entropy_str_max = float('-inf')
        entropy_str_min = float('inf')

        for idx in self.maj_ind:
            maj_count = len(set(neigh_ind[idx]).intersection(self.maj_ind))
            min_count = self.k - maj_count
            maj_con = float(maj_count) / float(self.k)
            min_con = float(min_count) / float(self.k)
            if min_count == 0:
                entropy_cer = 0
            else:
                entropy_cer = -maj_con * np.log(maj_con) - min_con * np.log(min_con)
                if maj_count < (self.k / 2):
                    entropy_cer = 2 * entropy_max - entropy_cer
            entropy_cer_max = entropy_cer if entropy_cer > entropy_cer_max else entropy_cer_max
            entropy_cer_min = entropy_cer if entropy_cer < entropy_cer_min else entropy_cer_min
            entropy_cer_list.append(entropy_cer)

            d_iq_sum = sum(neigh_dist[idx])
            log_list = [(d_iq / d_iq_sum) * np.log(d_iq / d_iq_sum) for d_iq in neigh_dist[idx]]
            entropy_str = 1.0 / (-sum(log_list))
            entropy_str_max = entropy_cer if entropy_str > entropy_str_max else entropy_str_max
            entropy_str_min = entropy_cer if entropy_str < entropy_str_min else entropy_str_min
            entropy_str_list.append(entropy_str)
        if entropy_cer_max == entropy_cer_min:
           normed_entropy_cer_list = [0] * len(entropy_cer_list)
        else:
            normed_entropy_cer_list = [(val - entropy_cer_min) / (entropy_cer_max - entropy_cer_min)
                                       for val in entropy_cer_list]
        if entropy_str_min == entropy_str_max:
            normed_entropy_str_list = [0] * len(entropy_str_list)
        else:
            normed_entropy_str_list = [(val - entropy_str_min) / (entropy_str_max - entropy_str_min)
                                       for val in entropy_str_list]
        entropy_all_list = np.sum([normed_entropy_cer_list, normed_entropy_str_list], axis=0)
        self.idx2entropy_dict = {}
        for entropy_idx, maj_idx in enumerate(self.maj_ind):
            self.idx2entropy_dict.update({maj_idx: entropy_all_list[entropy_idx]})

    def RandomUndersampling(self):
        rus = RandomUnderSampler(random_state=0)
        X_resampled, y_resampled = rus.fit_resample(self.X.copy(), self.y.copy())
        return X_resampled, y_resampled


if __name__ == '__main__':
    data = np.array([[0, 0.356, 0.468],
                     [0, 0.374, 0.498],
                     [0, 0.359, 0.460],
                     [0, 0.382, 0.476],
                     [1, 0.342, 0.48],
                     [0, 0.366, 0.51],
                     [0, 0.426, 0.454],
                     [1, 0.432, 0.48],
                     [0, 0.416, 0.464],
                     [1, 0.438, 0.44],
                     [1, 0.444, 0.464]
                     ])
    test = np.array([[0, 0.356, 0.494],
                     [0, 0.336, 0.508],
                     [1, 0.336, 0.464],
                     [1, 0.444, 0.464]
                     ])

    X = data[:, 1:]
    y = data[:, 0]
    test_X = test[:, 1:]
    test_y = test[:, 0]
    model = ECUBoostRF(L=10, lamda=0.1, k=5, T=10)
    model.fit(X, y)
    print(model.predict_proba(test_X))
    print(model.predict(test_X))

