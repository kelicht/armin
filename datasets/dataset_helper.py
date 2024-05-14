import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASETS = ['b', 'e', 'f', 'g', 's', 'w']

DATASET_NAMES = {
    'b': 'bank',
    'e': 'employeeattrition',
    'f': 'fico',
    'g': 'givemecredit',
    's': 'student',
    'w': 'winequality',
}

DATASET_FULLNAMES = {
    'b': 'Bank',
    'e': 'Attrition',
    'f': 'FICO',
    'g': 'GiveMeCredit',
    's': 'Student',
    'w': 'WineQuality',
}

TARGET_NAMES = {
    'b': 'subscribe',
    'e': 'Attrition',
    'f': 'RiskPerformance',
    'g': 'SeriousDlqin2yrs',
    's': 'Grade',
    'w': 'Quality',
}

FEATURE_TYPES = {
    'b': ['I']*6 + ['B']*29,
    'e': ['I']*5 + ['B'] + ['I']*5 + ['B','I','B'] + ['I']*9 + ['B']*21,
    'f': ['I']*23,
    'g': ['C', 'I', 'I', 'C'] + ['I'] * 6, 
    's': ['I']*6 + ['B']*8 + ['I']*7 + ['B']*27,
    'w': ['C']*5 + ['I']*2 + ['C']*4 + ['B'],
}

FEATURE_CONSTRAINTS = {
    'b': ['F']*2 + ['', 'I', '', 'I'] + ['F']*22 + ['']*7,
    'e': ['F', '', 'F', 'F', '', 'F'] + ['']*4 + ['F'] + ['']*8 + ['I']*4 + ['']*3 + ['F']*6 + ['']*9 + ['F']*3,
    'f': ['']*23,
    'g': ['', 'F'] + ['']*7 + ['F'], 
    's': ['F']*4 + [''] + ['F'] + ['']*14 + ['F']*28,
    'w': ['']*11 + ['F'],
}

FEATURE_CATEGORIES = {
    'b': [list(range(9, 21)), list(range(21, 24)), list(range(24, 28)), list(range(28, 31)), list(range(31, 35))],
    'e': [list(range(23, 26)), list(range(26, 32)), list(range(32, 41)), list(range(41, 44))],
    'f': [],
    'g': [], 
    's': [list(range(21,23)), list(range(23,25)), list(range(25,27)), list(range(27,29)), list(range(29,31)), list(range(31,36)), list(range(36,41)), list(range(41,45)), list(range(45,48))],
    'w': [],
}

TARGET_LABELS = {
    'b': ['Yes', 'No'],
    'e': ['No', 'Yes'],
    'f': ['Good', 'Bad'],
    'g': ['No', 'Yes'], 
    's': ['>10', '<=10'],
    'w': ['>5', '<=5'],
}



class Dataset():
    def __init__(self, dataset='g', drop_nan=True):
        """ A helper class for handling sample datasets.
        
        Parameters
        ----------
        dataset : {'b', 'e', 'f', 'g', 's', 'w'}, default='g'
            The dataset to be read. 
            - 'b': Bank
            - 'e': IBM Employee Attrition
            - 'f': FICO
            - 'g': Give Me Credit
            - 's': Student Performance
            - 'w': Wine Quality

        drop_nan : boolean, default=True
            Whethe to drop rows with missing values. 
        """

        self.df = pd.read_csv(CURRENT_DIR+'/{}.csv'.format(DATASET_NAMES[dataset]))
        if drop_nan: self.df = self.df.dropna()
        self.y = self.df[TARGET_NAMES[dataset]].values
        self.X = self.df.drop([TARGET_NAMES[dataset]], axis=1).values.astype(float)
        self.dataset_name = DATASET_NAMES[dataset]
        self.target_name = TARGET_NAMES[dataset]
        self.feature_names = list(self.df.drop([TARGET_NAMES[dataset]], axis=1).columns)
        self.feature_types = FEATURE_TYPES[dataset]
        self.feature_constraints = FEATURE_CONSTRAINTS[dataset]
        self.feature_categories = FEATURE_CATEGORIES[dataset]
        self.target_labels = TARGET_LABELS[dataset]
        
        self.params = {
            'feature_names': self.feature_names,  
            'feature_types': self.feature_types, 
            'feature_constraints': self.feature_constraints, 
            'feature_categories': self.feature_categories, 
            'target_name': self.target_name, 
            'target_labels': self.target_labels,
        }

    def get_dataset(self, split=False, test_size=0.25):
        """ Get the input samples X and output labels y of the dataset.

        Parameters
        ----------
        split : bool, default=False
            Whether to split the dataset into training and test samples. 
        
        test_size : float, default=0.25. 
            The ratio of the test size when train-test splitting. 

        Returns
        -------
        ret : tuple
            The pair of X and y. 
        """

        if split:
            X_tr, X_ts, y_tr, y_ts = train_test_split(self.X, self.y, test_size=test_size, stratify=self.y)
            return X_tr, X_ts, y_tr, y_ts
        else:
            return self.X, self.y
    
    def get_details(self):
        """ Get the details of the dataset.

        Returns
        -------
        details : pd.DataFrame
            The details on each feature in the dataset. 
        """

        features = self.feature_names
        types = ['Binary' if t == 'B' else ('Integer' if t == 'I' else 'Real') for t in self.feature_types]
        mins, maxs = self.X.min(axis=0), self.X.max(axis=0)
        immutables = ['Yes' if c == 'F' else 'No' for c in self.feature_constraints]
        constraints = ['Fix' if c == 'F' else ('Increasing only' if c == 'I' else ('Decreasing only' if c == 'D' else 'Nothing')) for c in self.feature_constraints]
        details = {
            'Feature': features,
            'Type': types, 
            'Min': mins,
            'Max': maxs, 
            'Immutable': immutables, 
            'Constraint': constraints
        }
        return pd.DataFrame(details)



def preprocess(df, target_name,  
               constrained_features={}, class_names=['Good', 'Bad'], prefix_sep=':', drop_nan=True):
    """ Get the input samples X, output labels y, target class y_target, and parameters from the passed classification dataset. 

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame object of the dataset. 
    
    target_name : str
        The column name of the target y in df. 
    
    target_class : int or str
        The desired class label. 
        
    constrained_features : dict, default={}
        The dictionary for passing the features with some constraints. 
        Its allowed key values are 'F', 'I', and 'D' representing fix, increasing only, and decreasing only, respectively. 
        Its each element is a list containing the feature names. 
        For example, the feature "Age" is allowed to be increased only, and "Gender" and "Married" must be fixed, 
        then constrained_features should be {'F': ['Gender', 'Married'], 'I': ['Age']}. 

    class_names : list of str, default=['Good', 'Bad'] 
        The class names of the target y. 
        
    prefix_sep : str, default=':'
        The separator string used when the categorical features are transformed through one-hot encoding by pandas.get_dummies. 

    drop_nan : boolean, default=True
        Whethe to drop rows with missing values. 

    Returns
    -------
    X : numpy.array of shape (n_samples, n_features)
        The input samples of the passed dataset. 

    y : numpy.array of shape (n_samples, )
        The output labels of the passed dataset. 
        
    params : dict   
        The parameters to be passed to the explainer class. 
        - feature_names : list of str, the list of the feature names. 
        - feature_types : list of str, the list of the feature types. 
            The type of each feature is 'C' (continuous value), 'I' (integer), or 'B' (binary). 
        - feature_constraints list of str, the list of the feature constraints. 
            The constraint of each feature is 'N' (no constraint), 'F' (fix), 'I' (increasing only), or 'D' (decreasing only). 
        - feature_categories : list, the list of the one-hot encoded categorical features. 
            Each element of the list is the list corresponding to one categorical feature, 
            and an inner list contains the feature indices after one-hot encoding of the original categorical feature. 
        - target_name : str, the column name of the target y. 
        - class_names : list of str, the class names of the target y. 

    """
   
    if drop_nan: df = df.dropna()
    y = df[target_name].values
    if not np.array_equal(np.unique(y), np.array([0, 1])):
        raise ValueError("TEARs currently supports only binary classification tasks (y in {0, 1}). ")

    df_processed = pd.get_dummies(df.drop(target_name, axis=1), prefix_sep=prefix_sep)
    X = df_processed.values.astype(np.float64)

    params = _get_feature_params(df_processed, constrained_features, prefix_sep)
    params['target_name'] = target_name
    params['class_names'] = class_names

    return X, y, params


def _get_feature_params(df_processed, constrained_features, prefix_sep):
    """ Get the dictionary of the parameters with respect to the features of the passed dataset. 
    """

    feature_names = df_processed.columns.values.tolist()

    feature_types = []
    for feature in feature_names:
        if df_processed[feature].dtype == float:
            feature_types.append('C')
        elif np.array_equal(np.array([0, 1]), np.sort(df_processed[feature].unique())):
            feature_types.append('B')
        else:
            feature_types.append('I')
            
    feature_constraints = []
    for feature in feature_names:
        if prefix_sep in feature:
            feature, _ = feature.split(prefix_sep)
        if 'F' in constrained_features and feature in constrained_features['F']:
            feature_constraints.append('F')
        elif 'I' in constrained_features and feature in constrained_features['I']:
            feature_constraints.append('I')
        elif 'D' in constrained_features and feature in constrained_features['D']:
            feature_constraints.append('D')
        else:
            feature_constraints.append('N')       
    
    feature_categories = []
    prefix = ''
    categories = []
    for d, feature in enumerate(feature_names):
        if prefix_sep not in feature:
            continue
        prefix_d, _ = feature.split(prefix_sep)
        if prefix == prefix_d:
            categories.append(d)
        else:
            if len(categories) > 0:
                feature_categories.append(categories)
            prefix = prefix_d
            categories = [d]
    if len(categories) > 0:
        feature_categories.append(categories)
        
    params = {
        'feature_names': feature_names,  
        'feature_types': feature_types, 
        'feature_constraints': feature_constraints, 
        'feature_categories': feature_categories, 
    }        
    return params
