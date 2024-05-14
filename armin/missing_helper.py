import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer


class MissingGenerator():
    def __init__(self, feature_types=[], feature_categories=[]):
        self.feature_types = feature_types
        self.feature_categories = feature_categories
        self.feature_categories_flatten = sum(feature_categories, [])
        self.feature_actuals = []
        d = 0; i = 0; 
        while(d < len(self.feature_types)):
            if self.feature_types[d]=='B' and d in self.feature_categories_flatten:
                self.feature_actuals.append(self.feature_categories[i])
                d += len(self.feature_categories[i])
                i += 1
            else:
                self.feature_actuals.append([d])
                d += 1
        self.n_feature_actuals = len(self.feature_actuals)

    def mask_instance(self, x, n_missing=1, m_=[]):
        n_missing = min(self.n_feature_actuals, n_missing)
        if len(m_)==0:
            i_missing = np.random.choice(np.arange(self.n_feature_actuals), size=n_missing, replace=False)
            m_ = sum([ self.feature_actuals[i] for i in i_missing ], [])
        x_masked = x.copy().astype(float)
        x_masked[m_] = np.nan
        return x_masked

    def mask_instances(self, X, n_missing=1, is_missing=None, m_=[]):
        X_masked = []
        if is_missing is None: 
            is_missing = [True] * X.shape[0]
        for n, x in enumerate(X): 
            if is_missing[n]:
                X_masked.append(self.mask_instance(x, n_missing=n_missing, m_=m_))
            else:
                X_masked.append(x)
        return np.array(X_masked)        

# class MissingGenerator


class SingleImputer(BaseEstimator, TransformerMixin):
    def __init__(self, imputer_type='mean', fill_value=None, max_iter=10, sample_posterior=False, n_neighbors=5, 
                 quantile=(0.01, 0.99), feature_types=[], feature_categories=[]):
        self.imputer_type = imputer_type
        self.fill_value = fill_value
        self.max_iter = max_iter
        self.sample_posterior = sample_posterior
        self.n_neighbors = n_neighbors
        self.quantile = quantile

        self.feature_types = feature_types
        self.feature_categories = feature_categories
        self.feature_categories_flatten = sum(feature_categories, [])
        self.feature_actuals = []
        d = 0; i = 0; 
        while(d < len(self.feature_types)):
            if self.feature_types[d]=='B' and d in self.feature_categories_flatten:
                self.feature_actuals.append(self.feature_categories[i])
                d += len(self.feature_categories[i])
                i += 1
            else:
                self.feature_actuals.append([d])
                d += 1
        self.n_feature_actuals = len(self.feature_actuals)

    def fit(self, X):
        self.n_features_ = X.shape[1]
        if self.imputer_type=='mice':
            self.imputer_ = IterativeImputer(max_iter=self.max_iter, sample_posterior=self.sample_posterior)
        elif self.imputer_type=='knn':
            self.imputer_ = KNNImputer(n_neighbors=self.n_neighbors)
        elif self.imputer_type=='tree':
            self.imputer_ = SimpleImputer().fit(X)
            self.imputer_.statistics_ = self.ub_
        else:
            self.imputer_ = SimpleImputer(strategy=self.imputer_type, fill_value=self.fill_value)
        self.imputer_ = self.imputer_.fit(X)
        return self

    def transform(self, X):
        is_single_instance = (X.shape==(self.n_features_, ))
        if is_single_instance: X = X.reshape(1, -1)
        X_isnan = np.isnan(X)
        X_transformed = self.imputer_.transform(X)
        X_imputed = X.copy()
        for x_isnan, x_transformed, x_imputed in zip(X_isnan, X_transformed, X_imputed):
            for f in self.feature_actuals:
                d = f[0]
                if not x_isnan[d]: continue
                if len(f)==1:
                    if self.feature_types[d] in ['B', 'I']: 
                        x_imputed[d] = x_transformed[d].round().astype(int)
                    else:
                        x_imputed[d] = x_transformed[d]
                else:
                    x_imputed[f] = 0
                    x_imputed[f[np.argmax(x_transformed[f])]] = 1
        return X_imputed[0] if is_single_instance else X_imputed

# class SingleImputer


class RandomImputer(BaseEstimator, TransformerMixin):
    def __init__(self, imputer_type='weighted', quantile=(0.01, 0.99), feature_types=[], feature_categories=[]):
        self.imputer_type = imputer_type
        self.quantile = quantile

        self.feature_types = feature_types
        self.feature_categories = feature_categories
        self.feature_categories_flatten = sum(feature_categories, [])
        self.feature_actuals = []
        d = 0; i = 0; 
        while(d < len(self.feature_types)):
            if self.feature_types[d]=='B' and d in self.feature_categories_flatten:
                self.feature_actuals.append(self.feature_categories[i])
                d += len(self.feature_categories[i])
                i += 1
            else:
                self.feature_actuals.append([d])
                d += 1
        self.n_feature_actuals = len(self.feature_actuals)

    def fit(self, X):
        self.n_features_ = X.shape[1]
        self.lb_ = np.nanquantile(X, self.quantile[0], axis=0)
        self.ub_ = np.nanquantile(X, self.quantile[1], axis=0)
        self.mean_ = np.nanmean(X, axis=0)
        self.std_ = np.nanstd(X, axis=0)
        return self

    def transform(self, X):
        is_single_instance = (X.shape==(self.n_features_, ))
        if(is_single_instance): X = X.reshape(1, -1)
        X_isnan = np.isnan(X)
        X_imputed = np.zeros_like(X)
        for n, x_isnan in enumerate(X_isnan):
            for f in self.feature_actuals:
                d = f[0]
                if not x_isnan[d]:
                    X_imputed[n, f] = X[n, f]
                    continue
                if len(f)==1:
                    if self.feature_types[d]=='B': 
                        if self.imputer_type=='uniform':
                            X_imputed[n, d] = np.random.randint(0, 2)
                        else:
                            X_imputed[n, d] = (np.random.uniform(0, 1) <= self.mean_[d]).astype(int)
                    else:
                        if self.imputer_type=='uniform':
                            if self.feature_types[d]=='I': 
                                X_imputed[n, d] = np.random.randint(self.lb_[d], self.ub_[d]+1)
                            else:
                                X_imputed[n, d] = np.random.uniform(self.lb_[d], self.ub_[d])
                        else:
                            if self.feature_types[d]=='I': 
                                X_imputed[n, d] = np.random.normal(self.mean_[d], self.std_[d]).astype(int)
                            else:
                                X_imputed[n, d] = np.random.normal(self.mean_[d], self.std_[d])
                else:
                    if self.imputer_type=='uniform':
                        D_1 = np.random.choice(f)
                    else:
                        D_1 = np.random.choice(f, p=self.mean_[f])
                    X_imputed[n, D_1] = 1
        for d in range(self.n_features_):
            X_imputed[X_imputed[:, d]<self.lb_[d], d] = self.lb_[d]
            X_imputed[X_imputed[:, d]>self.ub_[d], d] = self.ub_[d]
        return X_imputed[0] if is_single_instance else X_imputed

# class RandomImputer


class MultipleImputer(BaseEstimator, TransformerMixin):
    def __init__(self, n_sampling=10, imputer_type='weighted', max_iter=10, quantile=(0.01, 0.99), feature_types=[], feature_categories=[]):
        self.n_sampling = n_sampling
        self.imputer_type = imputer_type
        self.max_iter = max_iter
        self.quantile = quantile
        self.feature_types = feature_types
        self.feature_categories = feature_categories

    def fit(self, X):
        self.n_features_ = X.shape[1]
        if self.imputer_type == 'mice':
            self.imputer_ = SingleImputer(imputer_type='mice', max_iter=self.max_iter, sample_posterior=True, quantile=self.quantile, 
                                          feature_types=self.feature_types, feature_categories=self.feature_categories)
        else:
            self.imputer_ = RandomImputer(self.imputer_type, self.quantile, self.feature_types, self.feature_categories)
        self.imputer_ = self.imputer_.fit(X)
        return self

    def transform(self, X):
        return self.imputer_.transform(X)
    
    def generate_imputations(self, x):
        X = np.tile(x, (self.n_sampling, 1))
        return self.transform(X)

# class MultipleImputer

