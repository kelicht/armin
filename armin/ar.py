import numpy as np
import time

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

import gurobipy as gp
import os
from contextlib import redirect_stdout

from armin._utils import flatten, LimeEstimator
from armin._actions import Action, FeasibleActions


class AlgorithmicRecourseExplainer():
    def __init__(self, 
                 clf, 
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

        self.clf_ = clf
        self.feasible_actions_ = FeasibleActions(clf, X, Y, feature_names, feature_types, feature_categories, feature_constraints, max_candidates, quantile, tol)
        self.lime_approximation_ = lime_approximation
        if(isinstance(clf, LogisticRegression)):
            self.coef_ = clf.coef_[0]
            self.intercept_ = clf.intercept_[0]
            self.T_ = len(clf.coef_[0])
            self.lime_approximation_ = False
            self.is_linear_ = True
        elif(lime_approximation):
            self.lime_ = LimeEstimator(clf, X, n_samples=lime_n_samples, feature_types=feature_types, feature_categories=feature_categories, alpha=lime_alpha)
            self.T_ = X.shape[1]
            self.is_linear_ = True
        elif(isinstance(clf, RandomForestClassifier)):
            self.T_ = clf.n_estimators
            self.coef_ = np.ones(self.T_) / self.T_
            self.intercept_ = -1 * 0.5
            self.L_ = self.feasible_actions_.L_
            self.H_ = self.feasible_actions_.H_
            self.is_linear_ = False
        elif(isinstance(clf, XGBClassifier)):
            self.T_ = clf.n_estimators
            self.coef_ = np.ones(self.T_)
            self.intercept_ = 0.0
            self.L_ = self.feasible_actions_.L_
            self.H_ = self.feasible_actions_.H_
            self.is_linear_ = False
        elif(isinstance(clf, MLPClassifier)):
            self.hidden_coef_ = clf.coefs_[0]; self.hidden_coef_[abs(self.hidden_coef_)<1e-13] = 0.0; 
            self.coef_ = clf.coefs_[1]; self.coef_[abs(self.coef_)<1e-13] = 0.0; 
            self.hidden_intercept_ = clf.intercepts_[0]; self.hidden_intercept_[abs(self.hidden_intercept_)<1e-13] = 0.0; 
            self.intercept_ = clf.intercepts_[1][0]; 
            self.T_ = clf.intercepts_[0].shape[0]
            self.is_linear_ = False
        else:
            self.lime_approximation_ = True
            self.lime_ = LimeEstimator(clf, X, n_samples=lime_n_samples, feature_types=feature_types, feature_categories=feature_categories, alpha=lime_alpha)
            self.T_ = X.shape[1]
            self.is_linear_ = True

        self.D_ = X.shape[1]
        self.feature_names_ = feature_names if len(feature_names)==X.shape[1] else ['x_{}'.format(d) for d in range(X.shape[1])]
        self.feature_types_ = feature_types if len(feature_types)==X.shape[1] else ['C' for d in range(X.shape[1])]
        self.feature_categories_ = feature_categories
        self.feature_categories_flatten_ = flatten(feature_categories)
        self.feature_constraints_ = feature_constraints if len(feature_constraints)==X.shape[1] else ['' for d in range(X.shape[1])]
        self.target_name_ = target_name
        self.target_labels_ = target_labels
        self.tol_ = tol

    def apply(self, x, a):
        return self.feasible_actions_.apply(x, a)

    def counterfactual(self, x, a):
        return self.feasible_actions_.counterfactual(x, a)

    def counterfactuals(self, X, a):
        return self.feasible_actions_.counterfactuals(X, a)

    def is_feasible(self, x, a):
        return self.feasible_actions_.is_feasible(x, a)

    def is_valid(self, x, a, y_target=0):
        return self.feasible_actions_.is_valid(x, a, y_target=y_target)

    def cost(self, x, a, cost_type='TLPS'):
        return self.feasible_actions_.cost(x, a, cost_type=cost_type)

    def getModel(self, X, y_target, immutable_features, max_change_num, cost_type, confidence):    
        N = X.shape[0]
        K = min(max_change_num, self.D_)

        s = time.perf_counter()
        A, C, = self.feasible_actions_.generateActions(X, cost_type=cost_type)

        self.lb_ = [np.min(A_d) for A_d in A]; self.ub_ = [np.max(A_d) for A_d in A]; 
        non_zeros = []
        for d in range(self.D_):
            non_zeros_d = [1]*len(A[d])
            for i in range(len(A[d])):
                if(d in flatten(self.feature_categories_)):
                    if(A[d][i]<=0):
                        non_zeros_d[i] = 0
                elif(A[d][i]==0):
                    non_zeros_d[i] = 0
            non_zeros.append(non_zeros_d)

        # modeling start
        s = time.perf_counter()
        with redirect_stdout(open(os.devnull, 'w')): model = gp.Model()
        def LinSum(Vars): return gp.LinExpr( [1]*len(Vars), Vars )

        # variables: action
        self.variables_ = {}
        act = [model.addVar(name='act_{:04d}'.format(d), vtype=gp.GRB.CONTINUOUS, lb=self.lb_[d], ub=self.ub_[d]) for d in range(self.D_)]; self.variables_['act'] = act; 
        pi = [[model.addVar(name='pi_{:04d}_{:04d}'.format(d,i), vtype=gp.GRB.BINARY) for i in range(len(A[d]))] for d in range(self.D_)]; self.variables_['pi'] = pi; 
        cost = [model.addVar(name='cost_{:04d}'.format(n), vtype=gp.GRB.CONTINUOUS, lb=0) for n in range(N)]; self.variables_['cost'] = cost; 
        invalid =[ model.addVar(name='invalid_{:04d}'.format(n), vtype=gp.GRB.BINARY) for n in range(N)]; self.variables_['invalid'] = invalid; 
 
        # variables and constants: base learner
        if(isinstance(self.clf_, LogisticRegression) or self.lime_approximation_):
            xi  =[ [model.addVar(name='xi_{:04d}_{:04d}'.format(n,d), vtype=gp.GRB.CONTINUOUS, lb=X[n,d]+self.lb_[d], ub=X[n,d]+self.ub_[d]) for d in range(self.D_)] for n in range(N)]; self.variables_['xi'] = xi; 
        elif(isinstance(self.clf_, RandomForestClassifier)):
            xi  =[[model.addVar(name='xi_{:04d}_{:04d}'.format(n,t), vtype=gp.GRB.CONTINUOUS, lb=0, ub=1) for t in range(self.T_)] for n in range(N)]; self.variables_['xi'] = xi; 
            phi  =[ [[model.addVar(name='phi_{:04d}_{:04d}_{:04d}'.format(n,t,l), vtype=gp.GRB.BINARY) for l in range(self.L_[t])] for t in range(self.T_)] for n in range(N)]; self.variables_['phi'] = phi; 
            I = self.feasible_actions_.I_
        elif(isinstance(self.clf_, XGBClassifier)):
            xi  =[[model.addVar(name='xi_{:04d}_{:04d}'.format(n,t), vtype=gp.GRB.CONTINUOUS, lb=-1e+2, ub=1e+2) for t in range(self.T_)] for n in range(N)]; self.variables_['xi'] = xi; 
            phi  =[ [[model.addVar(name='phi_{:04d}_{:04d}_{:04d}'.format(n,t,l), vtype=gp.GRB.BINARY) for l in range(self.L_[t])] for t in range(self.T_)] for n in range(N)]; self.variables_['phi'] = phi; 
            I = self.feasible_actions_.I_
        elif(isinstance(self.clf_, MLPClassifier)):
            xi  = [[model.addVar(name='xi_{:04d}'.format(n,t), vtype=gp.GRB.CONTINUOUS, lb=0) for t in range(self.T_)] for n in range(N)]; self.variables_['xi'] = xi; 
            bxi  = [[model.addVar(name='bxi_{:04d}_{:04d}'.format(n,t), vtype=gp.GRB.CONTINUOUS, lb=0) for t in range(self.T_)] for n in range(N)]; self.variables_['bxi'] = bxi; 
            nu  =[[model.addVar(name='nu_{:04d}_{:04d}'.format(n,t), vtype=gp.GRB.BINARY) for t in range(self.T_)] for n in range(N)]; self.variables_['nu'] = nu; 
            M_bar, M = np.zeros(self.T_), np.zeros(self.T_)
            for t, w in enumerate(self.hidden_coef_.T):
                M_bar[t] += np.sum([min(w[d]*self.ub_[d], w[d]*self.lb_[d]) for d in range(self.D_)])
                M[t] += np.sum([max(w[d]*self.ub_[d], w[d]*self.lb_[d]) for d in range(self.D_)])

        # objective function: sum_{x \in X} C(a | x)
        model.setObjective(LinSum(cost), gp.GRB.MINIMIZE)
        model.addConstr(LinSum(cost) >= 0, name='C_basic_nonnegative_cost')

        # constraint: sum_{n} (1 - invalid_{n}) >= N * confidence
        model.addConstr(LinSum(invalid) <= int(N * (1 - confidence)), name='C_basic_confidence')

        # constraint: sum_{i} pi_{d,i} == 1
        for d in range(self.D_): model.addConstr(LinSum(pi[d]) == 1, name='C_basic_pi_{:04d}'.format(d))

        # constraint: a_d = sum_{i} a_{d,i} pi_{d,i}
        for d in range(self.D_): model.addConstr(act[d] - gp.LinExpr(A[d], pi[d]) == 0, name='C_basic_act_{:04d}'.format(d))

        # constraint: sum_{d} sum_{i} pi_{d,i} <= K
        if(K>=1): model.addConstr(gp.LinExpr(flatten(non_zeros), flatten(pi)) <= K, name='C_basic_sparsity')

        # constraint: sum_{i} pi_{d,i} == 0 if feature d is immutable
        if(len(immutable_features)>0): 
            for d in immutable_features:
                model.addConstr(gp.LinExpr(non_zeros[d], pi[d]) == 0, name='C_basic_immutable_{:04d}'.format(d))

        # constraint: sum_{d in G} a_d = 0
        for i, G in enumerate(self.feature_categories_): model.addConstr(LinSum([act[d] for d in G]) == 0, name='C_basic_category_{:04d}'.format(i))

        for n in range(N):
            # constraint: C(a | x) = sum_{d} sum_{i} c_{d,i} pi_{d,i}
            if(cost_type=='MPS'):
                for d in range(self.D_):
                    if((d in flatten(self.feature_categories_) and np.min(A[d])<0) or self.feature_constraints_[d]=='FIX'): continue
                    model.addConstr(cost[n] - gp.LinExpr(C[n][d], pi[d]) >= 0, name='C_{:04d}_cost_{:04d}'.format(n,d))
            else:
                model.addConstr(cost[n] - gp.LinExpr(flatten(C[n]), flatten(pi)) == 0, name='C_{:04d}_cost'.format(n))

            # constraint: invalid = I[h(x+a)!=h(x)]
            if(self.lime_approximation_):
                self.coef_, self.intercept_ = self.lime_.approximate(X[n])
                M_min=-1e+4; M_max=1e+4; 
            else:
                if(isinstance(self.clf_, RandomForestClassifier)):
                    M_min=-1.0; M_max=1.0; 
                else:
                    M_min=-1e+8; M_max=1e+8; 
            if(y_target == 1):
                # constraint: sum_{d} w_t xi_{n,t} + b >= M_min * invalid
                model.addConstr(gp.LinExpr(self.coef_, xi[n]) - M_min * invalid[n] >= - self.intercept_ + 1e-4, name='C_{:04d}_loss_ge'.format(n))
                # constraint: sum_{d} w_t xi_{n,t} + b <= M_max * (1-invalid)
                model.addConstr(gp.LinExpr(self.coef_, xi[n]) + M_max * invalid[n] <= M_max - self.intercept_, name='C_{:04d}_loss_le'.format(n))
            else:
                # constraint: sum_{d} w_t xi_{n,t} + b <= M_max * invalid
                model.addConstr(gp.LinExpr(self.coef_, xi[n]) - M_max * invalid[n] <= - self.intercept_ - 1e-4, name='C_{:04d}_loss_le'.format(n))
                # constraint: sum_{d} w_t xi_{n,t} + b >= M_min * (1-invalid)
                model.addConstr(gp.LinExpr(self.coef_, xi[n]) + M_min * invalid[n] >= M_min - self.intercept_, name='C_{:04d}_loss_ge'.format(n))

            # constraint (Linear model): xi_d = x_d + a_d
            if(isinstance(self.clf_, LogisticRegression) or self.lime_approximation_):
                # constraint: xi_d = x_d + a_d
                for d in range(self.D_): 
                    model.addConstr(xi[n][d] - act[d] == X[n,d], name='C_{:04d}_linear_{:04d}'.format(n,d))

            # constraints (Tree Ensemble):
            elif(isinstance(self.clf_, (RandomForestClassifier, XGBClassifier))):
                for t in range(self.T_):
                    # constraint: sum_{l} phi_{t,l} = 1
                    model.addConstr(LinSum(phi[n][t]) == 1, name='C_{:04d}_forest_leaf_{:04d}'.format(n,t))
                    # constraint: xi_t = sum_{l} h_{t,l} phi_{t,l}
                    model.addConstr(xi[n][t] - gp.LinExpr(self.H_[t], phi[n][t]) == 0, name='C_{:04d}_forest_{:04d}'.format(n,t))
                    # constraint: D * phi_{t,l} <= sum_{d} sum_{i in I_{t,l,d}} pi_{d,i}
                    for l in range(self.L_[t]):
                        p = self.feasible_actions_.ancestors_[t][l]
                        model.addConstr(len(p) * phi[n][t][l] - gp.LinExpr(flatten([I[n][t][l][d] for d in p]), flatten([pi[d] for d in p])) <= 0, name='C_{:04d}_forest_decision_{:04d}_{:04d}'.format(n,t,l))

            # constraints (Multi-Layer Perceptoron):
            elif(isinstance(self.clf_, MLPClassifier)):
                M_bar_n = -1 * (X[n].dot(self.hidden_coef_)+self.hidden_intercept_ + M_bar); M_n = X[n].dot(self.hidden_coef_)+self.hidden_intercept_ + M;
                M_bar_n[M_bar_n<0] = 0.0; M_n[M_n<0] = 0.0; 
                M_bar_n[M_bar_n>0] += self.tol_; M_n[M_n>0] += self.tol_; 

                for t in range(self.T_): 
                    ## constraint: xi_t <= M_t nu_t
                    ## constraint: bxi_t <= M_bar_t (1-nu_t)
                    model.addConstr(xi[n][t] - M_n[t] * nu[n][t] <= 0, name='C_{:04d}_mlp_pos_{:04d}'.format(n,t))
                    model.addConstr(bxi[n][t] + M_bar_n[t] * nu[n][t] <= M_bar_n[t], name='C_{:04d}_mlp_neg_{:04d}'.format(n,t))

                    ## constraint: xi_t = bxi_t + sum_{d} w_{t,d} (x_d + a_d) + b_t
                    model.addConstr(xi[n][t] - bxi[n][t] - gp.LinExpr(self.hidden_coef_.T[t], act) == X[n].dot(self.hidden_coef_.T[t]) + self.hidden_intercept_[t], name='C_{:04d}_mlp_{:04d}'.format(n,t))

        self.actions_ = A
        self.time_modeling_ = time.perf_counter()-s
        return model


    def solveModel(self, model, X, y_target, y_init, cost_type, time_limit, log_name, verbose):

        if(len(log_name)!=0): model.write(log_name+'.lp')
        model.params.outputflag = int(verbose)
        model.params.timelimit = time_limit
        model.optimize()
        t = model.runtime

        try:
            a = np.array([ np.sum([ self.actions_[d][i] * round(self.variables_['pi'][d][i].X)  for i in range(len(self.actions_[d])) ]) for d in range(self.D_) ])
            infeasible = False
        except AttributeError:
            infeasible = True

        action_dicts = []
        if infeasible:
            if verbose: model.write('infeasible.lp')
            y_prob = self.clf_.predict_proba( X )
            for n, x in enumerate(X):
                action_dicts.append( {'solved': False, 
                                      'action': np.zeros(self.D_), 
                                      'cost': 0.0, 
                                      'loss': 1,
                                      'valid': False,
                                      'feasible': True,
                                      'instance': x,
                                      'probability': dict(zip(self.target_labels_, y_prob[n])),
                                      'probability_target': y_prob[n][y_target],
                                      'cost_type': cost_type,
                                      'y_target': y_target,
                                      'y_init': y_init,
                                      'time': t} )
        else:
            y_prob = self.clf_.predict_proba( self.counterfactuals(X, a) )
            for n, x in enumerate(X):
                action_dict = {'solved': True,
                               'action': self.apply(x, a), 
                               'cost': self.cost(x, a, cost_type=cost_type), 
                               'loss': self.variables_['invalid'][n].X,
                               'valid': self.is_valid(x, a, y_target=y_target),
                               'feasible': self.is_feasible(x, a),
                               'instance': x,
                               'probability': dict(zip(self.target_labels_, y_prob[n])),
                               'probability_target': y_prob[n][y_target],
                               'cost_type': cost_type,
                               'y_target': y_target,
                               'y_init': y_init,
                               'time': t}
                action_dicts.append(action_dict)

        return action_dicts



    def extract(self, X, y_target=0, 
                max_change_num=4, cost_type='TLPS',  immutable_features=[], confidence=1.0, time_limit=180, log_name='', verbose=False):

        is_single_instance =  (X.shape==(self.D_,))
        if(is_single_instance): X = X.reshape(1, -1)
        y_init = self.clf_.predict(X)[0]
        self.feasible_actions_.immutable_features_ = immutable_features

        model = self.getModel(X, y_target, immutable_features, max_change_num, cost_type, confidence)
        action_dicts = self.solveModel(model, X, y_target, y_init, cost_type, time_limit, log_name, verbose)

        return action_dicts[0] if is_single_instance else action_dicts

    def updateActionDicts(self, x, action_dicts):
        is_single_action = (type(action_dicts)==dict)
        if(is_single_action): action_dicts = [ action_dicts ]
        for action_dict in action_dicts:
            a = self.apply(x, action_dict['action'])
            action_dict['action'] = a
            action_dict['cost'] = self.cost(x, a, cost_type=action_dict['cost_type'])
            action_dict['loss'] = 1-int(self.is_valid(x, a, y_target=action_dict['y_target']))
            action_dict['valid'] = self.is_valid(x, a, y_target=action_dict['y_target'])
            action_dict['feasible'] = self.is_feasible(x, a)
            action_dict['instance'] = x
            action_dict['probability'] = dict(zip(self.target_labels_, self.clf_.predict_proba(self.counterfactual(x, a).reshape(1,-1))[0]))
            action_dict['probability_target'] = action_dict['probability'][self.target_labels_[action_dict['y_target']]]
        return action_dicts[0] if is_single_action else action_dicts

    def getActionObject(self, action_dicts, keys=['feasible','valid','cost','probability','time'], print_instance=False, print_features=[]):
        is_single_action = (type(action_dicts)==dict)
        if(is_single_action): action_dicts = [ action_dicts ]
        ret = []
        for action_dict in action_dicts:
            action = Action(action_dict['instance'], 
                            action_dict['action'], 
                            scores={key: action_dict[key] for key in keys},
                            target_name=self.target_name_, 
                            target_labels=self.target_labels_, 
                            label_before=int(self.clf_.predict(action_dict['instance'].reshape(1,-1))[0]), 
                            label_after=int(self.clf_.predict((action_dict['instance']+action_dict['action']).reshape(1,-1))[0]), 
                            feature_names=self.feature_names_, 
                            feature_types=self.feature_types_, 
                            feature_categories=self.feature_categories_,
                            print_instance=print_instance,
                            print_features=print_features)
            ret.append(action)
        return ret[0] if is_single_action else ret
        
# class AlgorithmicRecourseExplainer

