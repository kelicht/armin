import numpy as np
import time
import gurobipy as gp

from armin.ar import AlgorithmicRecourseExplainer


class ImputationAlgorithmicRecourseExplainer(AlgorithmicRecourseExplainer):
    def __init__(self, 
                 clf, 
                 imputer,
                 X, 
                 Y=[],
                 feature_names=[], 
                 feature_types=[], 
                 feature_categories=[], 
                 feature_constraints=[], 
                 max_candidates=100, 
                 quantile=(0.0, 1.0), 
                 tol=1e-6,
                 target_name='Output', 
                 target_labels = ['Good','Bad'], 
                 lime_approximation=False, 
                 lime_n_samples=5000, 
                 lime_alpha=1.0,
        ):

        super().__init__(clf, X, Y, 
                         feature_names, feature_types, feature_categories, feature_constraints, max_candidates, 
                         quantile, tol, target_name, target_labels, lime_approximation, lime_n_samples, lime_alpha)
    
        self.imputer_ = imputer
    
    def extract(self, X_missing, y_target=0, 
                max_change_num=4, cost_type='TLPS', immutable_features=[], confidence=1.0, time_limit=180, log_name='', verbose=False):
        
        X = self.imputer_.transform(X_missing)
        return super().extract(X, y_target, max_change_num, cost_type, immutable_features, confidence, time_limit, log_name, verbose)

# class ImputationAlgorithmicRecourseExplainer


class RobustAlgorithmicRecourseExplainer(AlgorithmicRecourseExplainer):
    def __init__(self, 
                 clf, 
                 imputer,
                 X, 
                 Y=[],
                 feature_names=[], 
                 feature_types=[], 
                 feature_categories=[], 
                 feature_constraints=[], 
                 max_candidates=100, 
                 quantile=(0.0, 1.0), 
                 tol=1e-6,
                 target_name='Output', 
                 target_labels = ['Good','Bad'], 
                 lime_approximation=False, 
                 lime_n_samples=5000, 
                 lime_alpha=1.0,
        ):

        super().__init__(clf, X, Y, 
                         feature_names, feature_types, feature_categories, feature_constraints, max_candidates, 
                         quantile, tol, target_name, target_labels, lime_approximation, lime_n_samples, lime_alpha)
    
        self.imputer_ = imputer
    
    def extract(self, x_missing, y_target=0, 
                max_change_num=4, cost_type='TLPS', immutable_features=[], time_limit=180, max_iter=100, log_name='', verbose=False):

        X_sample = self.imputer_.generate_imputations(x_missing)
        n_samples = X_sample.shape[0]
        
        a = np.zeros(self.D_)
        I = list(range(1, n_samples))
        J = [0]
        solved = False
        
        s = time.time() 
        for iter in range(max_iter): 
            model = self.getModel(X_sample[J], y_target, immutable_features, max_change_num, cost_type, 1.0)
            if(len(log_name)!=0): model.write(log_name+'.lp')

            model.params.outputflag = int(verbose) 
            model.params.timelimit = time_limit
            model.optimize()

            try:
                a = np.array([ np.sum([ self.actions_[d][i] * round(self.variables_['pi'][d][i].X)  for i in range(len(self.actions_[d])) ]) for d in range(self.D_) ])
                infeasible = False
            except AttributeError:
                infeasible = True

            if infeasible:
                J.pop()
                if len(J)==0: break
            else: 
                solved = True
                y_prob = self.clf_.predict_proba(self.counterfactuals(X_sample, a))[:, y_target]
            if len(I)==0: break
            ii = np.argmin(y_prob[I])
            if y_prob[I[ii]]>=0.5: break
            J.append(I.pop(ii))
        t = time.time() - s

        x = X_sample[0]
        y_init = self.clf_.predict(X_sample)[0]
        y_after = self.clf_.predict( self.counterfactuals(X_sample, a) )[0]
        y_prob = self.clf_.predict_proba( self.counterfactuals(X_sample, a) )[0]
        action_dict = {'solved': solved,
                        'action': self.apply(x, a), 
                        'cost': self.cost(x, a, cost_type=cost_type), 
                        'loss': int(y_after!=y_target),
                        'valid': self.is_valid(x, a, y_target=y_target),
                        'feasible': self.is_feasible(x, a),
                        'instance': x,
                        'probability': dict(zip(self.target_labels_, y_prob)),
                        'probability_target': y_prob[y_target],
                        'cost_type': cost_type,
                        'y_target': y_target,
                        'y_init': y_init,
                        'time': t}
        return action_dict

# class RobustAlgorithmicRecourseExplainer