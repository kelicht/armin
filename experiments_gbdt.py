import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from datasets import Dataset
from armin import AlgorithmicRecourseExplainer
from armin.missing_helper import MissingGenerator
from armin._utils import sign_agreement



def exp_gbdt(N=3, dataset='f', max_n_missing=1, max_change_num=3, cost_type='TLPS', res_name='', res_suffix=''):
    np.random.seed(0)
    model = 'X'

    clf = XGBClassifier(n_estimators=50, max_leaves=8, tree_method='hist')
    D = Dataset(dataset=dataset)
    X_tr, X_ts, y_tr, y_ts = D.get_dataset(split=True, test_size=0.25)
    clf = clf.fit(X_tr, y_tr)
    X = X_ts[clf.predict(X_ts)==1][:N]
    N = X.shape[0]

    ar = AlgorithmicRecourseExplainer(clf, X_tr, Y=y_tr,
                                 feature_names=D.feature_names, feature_types=D.feature_types, feature_categories=D.feature_categories, 
                                 feature_constraints=D.feature_constraints, quantile=(dataset=='k'), target_name=D.target_name, target_labels=D.class_names)
    mg = MissingGenerator(feature_types=D.feature_types, feature_categories=D.feature_categories)

    keys = ['n_missing', 'method', 'feasible', 'valid', 'cost', 'relative_cost', 'time', 'probability_target', 'sign_agreement', 'y_init']
    res = dict([(key, []) for key in keys])
    for n in range(N):
        print('# Instance', n+1)

        print('## Optimal action without missing')
        action = ar.extract(X[n], y_target=0, max_change_num=max_change_num, cost_type=cost_type)
        if not action['solved']: continue
        a = action['action']; c = action['cost']; 
        action['sign_agreement'] = sign_agreement(action['action'], a); action['relative_cost'] = action['cost'];  
        print(ar.getActionObject(action, print_instance=True))
        res['n_missing'].append(0); res['method'].append('complete'); 
        for key in keys[2:]: res[key].append(action[key])

        for n_missing in range(1, max_n_missing+1):
            X_missing = mg.mask_instances(X, n_missing=n_missing)
            missing_features = np.where(np.isnan(X_missing[n]))[0]

            print('## Optimal action if missing')
            action = ar.extract(X_missing[n], y_target=0, max_change_num=max_change_num, cost_type=cost_type, immutable_features=missing_features)
            action = ar.updateActionDicts(X[n], action)
            action['sign_agreement'] = sign_agreement(action['action'], a); action['relative_cost'] = action['cost'] / c; 
            print(ar.getActionObject(action, print_instance=True, print_features=missing_features))
            res['n_missing'].append(n_missing); res['method'].append('incomplete');  
            for key in keys[2:]: res[key].append(action[key])

    if len(res_name)==0: res_name = 'gbdt_{}_{}'.format(dataset, cost_type)
    if len(res_suffix)>0: res_name = res_name + '_' + res_suffix
    pd.DataFrame(res).to_csv('./res/{}/{}.csv'.format(model, res_name), index=False)



if __name__ == '__main__':
    N = 100
    max_n_missing = 4
    max_change_num = 4

    for dataset in ['f', 'e', 'w', 's']:
        for cost_type in ['TLPS', 'MAD']:
            exp_gbdt(N=N, dataset=dataset, max_n_missing=max_n_missing, max_change_num=max_change_num, cost_type=cost_type)