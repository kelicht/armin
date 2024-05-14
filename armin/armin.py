import numpy as np
import gurobipy as gp
import time
from armin.ar import AlgorithmicRecourseExplainer

def LinSum(Vars): return gp.LinExpr( [1]*len(Vars), Vars )



class ArminExplainer(AlgorithmicRecourseExplainer):
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


    def _extract_by_subsampling(self, x_missing, y_target=0, max_change_num=4, cost_type='TLPS', immutable_features=[], confidence=0.5, time_limit=180, n_subsampling=10, max_iter=10, log_name='', verbose=False):

        X_sample = self.imputer_.generate_imputations(x_missing)
        n_samples = X_sample.shape[0]
        
        A = np.zeros((max_iter, self.D_))
        C = np.zeros(max_iter)
        V = np.zeros(max_iter)
        solved = False
        
        s = time.time() 
        for iter in range(max_iter): 
            J = np.random.choice(n_samples, n_subsampling, replace=False)
            model = self.getModel(X_sample[J], y_target, immutable_features, max_change_num, cost_type, confidence)
            if(len(log_name)!=0): model.write(log_name+'.lp')

            model.params.outputflag = int(verbose) 
            model.params.timelimit = time_limit
            model.optimize()

            try:
                A[iter] = np.array([ np.sum([ self.actions_[d][i] * round(self.variables_['pi'][d][i].X)  for i in range(len(self.actions_[d])) ]) for d in range(self.D_) ])
                V[iter] = self.clf_.predict_proba(self.counterfactuals(X_sample, A[iter]))[:, y_target].mean()
                C[iter] = model.ObjVal
            except AttributeError:
                V[iter] = self.clf_.predict_proba(X_sample)[:, y_target].mean()

            if V[iter] >= confidence:
                solved = True

        t = time.time() - s
        if solved:
            iter_valid = np.where(V >= confidence)[0]
            best_iter = iter_valid[np.argmin(C[iter_valid])]
        else:
            best_iter = np.argmax(V)
        a = A[best_iter]

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

    def extract(self, x_missing, y_target=0, algorithm='auto', 
                max_change_num=4, cost_type='TLPS', immutable_features=[], confidence=0.5, time_limit=180, n_subsampling=10, max_iter=10, log_name='', verbose=False):

        if algorithm == 'auto':
            if self.is_linear_:
                algorithm = 'exact'
            else:
                algorithm = 'heuristic'

        if algorithm == 'exact':
            X_sample = self.imputer_.generate_imputations(x_missing)
            actions_dicts = super().extract(X_sample, y_target, max_change_num, cost_type, immutable_features, confidence, time_limit, log_name, verbose)
            return actions_dicts[0]

        else:
            action_dict = self._extract_by_subsampling(x_missing, y_target, max_change_num, cost_type, immutable_features, confidence, time_limit, n_subsampling, max_iter, log_name, verbose)
            return action_dict
    
    def confidence_path(self, x_missing, stepsize=1, y_target=0, 
                        max_change_num=4, cost_type='TLPS', immutable_features=[], time_limit=180, log_name='', verbose=False):
        
        X_sample = self.imputer_.generate_imputations(x_missing)
        N = X_sample.shape[0]
        rho = 1 / N
        path = []

        y_init = self.clf_.predict(X_sample)[0]
        self.feasible_actions_.immutable_features_ = immutable_features

        model = self.getModel(X_sample, y_target, immutable_features, max_change_num, cost_type, rho)
        action_dicts = self.solveModel(model, X_sample, y_target, y_init, cost_type, time_limit, log_name, verbose)
        path.append((rho, action_dicts[0]))
        rho = (stepsize + np.sum([action_dict['valid'] for action_dict in action_dicts])) / X_sample.shape[0]

        while rho <= 1:
            model.remove(model.getConstrByName('C_basic_confidence'))
            invalid = [model.getVarByName('invalid_{:04d}'.format(n)) for n in range(N)]
            model.addConstr(LinSum(invalid) <= int(N * (1 - rho)), name='C_basic_confidence')
            action_dicts = self.solveModel(model, X_sample, y_target, y_init, cost_type, time_limit, log_name, verbose)
            if action_dicts[0]['solved']:
                path.append((rho, action_dicts[0]))
                rho = (stepsize + (N - np.sum([int(i.X) for i in invalid]))) / X_sample.shape[0]
            else:
                break

        return path

