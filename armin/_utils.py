import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde as kde
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels



def flatten(x): return sum(x, [])

def supp(a, tol=1e-8): return np.where(abs(a)>tol)[0]

def sign_agreement(a_1, a_2): 
    k = max(supp(a_1).shape[0], supp(a_2).shape[0])
    if k == 0:
        return 1.0
    else:
        return np.sum((a_1 * a_2) > 0) /  k

def CumulativeDistributionFunction(x_d, X_d, l_buff=1e-6, r_buff=1e-6):
    kde_estimator = kde(X_d)
    pdf = kde_estimator(x_d)
    cdf_raw = np.cumsum(pdf)
    total = cdf_raw[-1] + l_buff + r_buff
    cdf = (l_buff + cdf_raw) / total
    percentile_ = interp1d(x=x_d, y=cdf, copy=False,fill_value=(l_buff,1.0-r_buff), bounds_error=False, assume_sorted=False)
    return percentile_

def parse_xgbtree(tree_info, node_count):
    """ Parse the XGBoost object into the numpy.array expression like Tree object of scikit-learn. 
    """
    feature = -2 * np.ones(node_count, dtype=np.int64)
    threshold = -2 * np.ones(node_count, dtype=np.float64)
    value = np.zeros(node_count, dtype=np.float64)
    children_left = -1 * np.ones(node_count, dtype=np.int64)
    children_right = -1 * np.ones(node_count, dtype=np.int64)

    # Traverse the tree for extracting the parameters of each node by BFS. 
    queue = [tree_info]
    while len(queue) > 0:
        node_info = queue.pop(0)
        j = node_info['nodeid']
        if 'leaf' in node_info:
            value[j] = node_info['leaf']
        else:
            feature[j] = int(node_info['split'].replace('f', ''))
            threshold[j] = node_info['split_condition']
            children_left[j] = node_info['yes']
            children_right[j] = node_info['no']
            children = node_info['children']
            queue.append(children[0])
            queue.append(children[1])

    ret = {
        'node_count': node_count,
        'feature': feature, 
        'threshold': threshold,
        'value': value,
        'children_left': children_left,
        'children_right': children_right,
    }
    return GBMTree(**ret)


class GBMTree():
    def __init__(self, node_count, feature, threshold, value, children_left, children_right):
        self.node_count = node_count
        self.feature = feature
        self.threshold = threshold
        self.value = value
        self.children_left = children_left
        self.children_right = children_right
        

class LimeEstimator():
    def __init__(self, mdl, X, n_samples=10000, feature_types=[], feature_categories=[], alpha=1.0):
        self.mdl_ = mdl
        self.mdl_local_ = Ridge(alpha=alpha)
        self.N_, self.D_ = X.shape
        self.n_samples_ = n_samples
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)

        self.feature_types_ = feature_types if len(feature_types)==self.D_ else ['C' for d in range(self.D_)]
        self.feature_category_ = feature_categories
        self.feature_category_flatten_ = flatten(feature_categories)
        self.feature_ordered_ = [d for d in range(self.D_) if feature_types[d]=='C' or feature_types[d]=='I']
        self.feature_binary_ = [d for d in range(self.D_) if feature_types[d]=='B' and d not in self.feature_category_flatten_]

    def getNeighbors(self, x):
        N_x = np.zeros([self.n_samples_, self.D_])
        for d in self.feature_ordered_:
            if(self.feature_types_[d]=='I'):
                N_x[:, d] = np.random.normal(x[d], self.std_[d], self.n_samples_).astype(int)
            else:
                N_x[:, d] = np.random.normal(x[d], self.std_[d], self.n_samples_)
        for d in self.feature_binary_:
            N_x[:, d] = (np.random.uniform(0, 1, self.n_samples_) <= self.mean_[d]).astype(int)
        for G in self.feature_category_:
            cats = np.random.choice(G, self.n_samples_, p=self.mean_[G])
            for n, d in enumerate(cats): N_x[n, d] = 1
        N_x = np.concatenate([x.reshape(1,-1), N_x], axis=0)
        return N_x

    def getWeights(self, x, N_x):
        distance = pairwise_distances(N_x/self.std_, (x/self.std_).reshape(1,-1)).reshape(-1)
        kernel_width = np.sqrt(self.D_) * .75
        weights = np.sqrt(np.exp(-(distance ** 2) / kernel_width ** 2))
        return weights

    def fit(self, x, target_label=None):
        N_x = self.getNeighbors(x)
        weights = self.getWeights(x, N_x)
        if(target_label is None): target_label = int(1-self.mdl_.predict(x.reshape(1, -1))[0])
        self.mdl_local_ = self.mdl_local_.fit(N_x, self.mdl_.predict_proba(N_x)[:, target_label], sample_weight=weights)
        self.offset_ = self.mdl_.predict_proba(x.reshape(1, -1))[0, target_label] - self.mdl_local_.predict(x.reshape(1, -1))[0]
        return self

    def approximate(self, x):
        self = self.fit(x)
        return self.mdl_local_.coef_, self.mdl_local_.intercept_+self.offset_-0.5

    def predict(self, X):
        return self.mdl_local_.predict(X)


