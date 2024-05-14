import numpy as np
import json
from scipy.stats import median_absolute_deviation as mad
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from armin._utils import CumulativeDistributionFunction, parse_xgbtree


ACTION_TYPES = ['B', 'I', 'C']
ACTION_CONSTRAINTS = ['', 'F', 'I', 'D']


class FeasibleActions():
    def __init__(self, clf, X, Y=[], 
                 feature_names=[], feature_types=[], feature_categories=[], feature_constraints=[], 
                 max_candidates=100, quantile=(0.0, 1.0), tol=1e-6):
        self.clf_ = clf
        self.X_ = X
        self.Y_ = Y
        self.N_, self.D_ = X.shape
        self.feature_names_ = feature_names if len(feature_names)==self.D_ else ['x_{}'.format(d) for d in range(self.D_)]
        self.feature_types_ = feature_types if len(feature_types)==self.D_ else ['C' for d in range(self.D_)]
        self.feature_categories_ = feature_categories
        self.feature_constraints_ = feature_constraints if len(feature_constraints)==self.D_ else ['' for d in range(self.D_)]
        self.max_candidates = max_candidates
        self.tol_ = tol

        self.X_lb_, self.X_ub_ = np.quantile(X, quantile[0], axis=0), np.quantile(X, quantile[1], axis=0)
        self.steps_ = [(self.X_ub_[d]-self.X_lb_[d])/(max_candidates-1) if self.feature_types_[d]=='C' else max(1, int((self.X_ub_[d]-self.X_lb_[d])/(max_candidates-1))) for d in range(self.D_)]
        self.grids_ = [np.arange(self.X_lb_[d], self.X_ub_[d]+self.steps_[d], self.steps_[d]) for d in range(self.D_)]

        self.x_ = None
        self.actions_ = None
        self.costs_ = None
        self.cost_type_ = None
        self.Q_ = None
        self.weights_ = None
        self.immutable_features_ = []

        self.is_forest_ = isinstance(clf, (RandomForestClassifier, XGBClassifier))
        if(self.is_forest_):
            self.T_ = clf.n_estimators
            if isinstance(clf, XGBClassifier):
                forest_info = [json.loads(tree_info) for tree_info in clf.get_booster().get_dump(dump_format='json')]
                node_counts = [2 * tree_info.count('leaf') - 1 for tree_info in clf.get_booster().get_dump(dump_format='json')]
                self.trees_ = [ parse_xgbtree(tree_info, node_count) for tree_info, node_count in zip(forest_info, node_counts) ]
            else:
                self.trees_ = [t.tree_ for t in clf.estimators_]
            self.leaves_ = [np.where(tree.feature==-2)[0]  for tree in self.trees_]
            self.L_ = [len(l) for l in self.leaves_]
            self.H_ = self.getForestLabels()
            self.ancestors_, self.regions_ = self.getForestRegions()        
            self.thresholds_ = self.getForestThresholds()
            self.I_ = None

    def getFeatureWeight(self, cost_type='uniform'):
        weights = np.ones(self.D_)
        if(cost_type=='MAD'):
            for d in range(self.D_):
                weight =  mad(self.X_[:,d])
                if(self.feature_types_[d]=='B' or abs(weight)<self.tol_):
                    weights[d] = (self.X_[:,d]*1.4826).std()
                else:
                    weights[d] = weight ** -1
        elif(cost_type=='PCC' and len(self.Y_)==self.N_):
            for d in range(self.D_):
                weights[d] = abs(np.corrcoef(self.X_[:, d], self.Y_)[0,1])
        elif(cost_type=='standard'):
            weights = np.std(self.X_, axis=0) ** -1
        elif(cost_type=='normalize'):
            weights = (self.X_.max(axis=0) - self.X_.min(axis=0)) ** -1
        elif(cost_type=='robust'):
            q25, q75 = np.percentile(self.X_, [0.25, 0.75], axis=0)
            for d in range(self.D_):
                if(q75[d]-q25[d]==0):
                    weights[d] = self.tol_ ** -1
                else:
                    weights = (q75[d]-q25) ** -1
        return weights

    def getMultiActionSet(self, xs, union=False, use_threshold=True):
        actions = []
        for d in range(self.D_):
            if(self.feature_constraints_[d]=='F' or d in self.immutable_features_ or self.steps_[d] < self.tol_):
                actions.append(np.array([ 0 ]))
            elif(self.feature_types_[d]=='B'):
                x_d = xs[0, d]
                if(union):
                    actions.append(np.array([ -1, 1, 0 ]))
                elif((xs[:, d]==x_d).all()):
                    if((self.feature_constraints_[d]=='I' and x_d==1) or (self.feature_constraints_[d]=='D' and x_d==0)):
                        actions.append(np.array([ 0 ]))
                    else:
                        actions.append(np.array([ 1-2*x_d, 0 ]))
                else:
                    actions.append(np.array([ 0 ]))
            else:
                x_min = np.max(xs[:, d]) if union else np.min(xs[:, d])
                x_max = np.min(xs[:, d]) if union else np.max(xs[:, d])
                if(self.is_forest_ and use_threshold):
                    A_d = np.array([])
                    for x in xs:
                        A_d = np.concatenate([A_d, self.thresholds_[d].astype(int)-x[d] if self.feature_types_[d]=='I' else self.thresholds_[d]-x[d]])
                    if isinstance(self.clf_, XGBClassifier):
                        A_d[A_d<0] -= self.tol_ if self.feature_types_[d]=='C' else 1
                        if(self.feature_types_[d]=='C'): A_d[A_d>0] += self.tol_
                    else:                    
                        A_d[A_d>0] += self.tol_ if self.feature_types_[d]=='C' else 1
                        if(self.feature_types_[d]=='C'): A_d[A_d<0] -= self.tol_
                    A_d = np.unique(A_d)
                    if(self.feature_constraints_[d]=='I'): 
                        A_d = np.extract(A_d>=0, A_d)
                    elif(self.feature_constraints_[d]=='D'): 
                        A_d = np.extract(A_d<=0, A_d)
                    A_d = np.extract(x_min+A_d>=self.X_lb_[d], A_d)
                    A_d = np.extract(x_max+A_d<=self.X_ub_[d], A_d)
                    if(A_d.shape[0]>self.max_candidates): A_d = A_d[np.linspace(0, A_d.shape[0], self.max_candidates, endpoint=False, dtype=int)]
                    if(0 not in A_d): A_d = np.append(A_d, 0)
                else:
                    if(self.feature_constraints_[d]=='I'):
                        start = self.steps_[d]
                        stop = self.X_ub_[d] + self.steps_[d] - x_max
                    elif(self.feature_constraints_[d]=='D'):
                        start = self.X_lb_[d] - x_min
                        stop = 0
                    else:
                        start = self.X_lb_[d] - x_min
                        stop = self.X_ub_[d] + self.steps_[d] - x_max
                    A_d = np.arange(start, stop, self.steps_[d])
                    A_d = np.extract(abs(A_d)>self.tol_, A_d)
                    if(len(A_d) > self.max_candidates): A_d = A_d[np.linspace(0, len(A_d)-1, self.max_candidates, dtype=int)]
                    A_d = np.append(A_d, 0)
                actions.append(A_d)
        return actions

    def getCostSet(self, x, cost_type='TLPS', p=1):
        costs = []
        if(cost_type=='TLPS' or cost_type=='MPS'):
            if(self.Q_==None): self.Q_ = [None if self.feature_constraints_[d]=='F' else CumulativeDistributionFunction(self.grids_[d], self.X_[:, d]) for d in range(self.D_)]
            for d in range(self.D_):
                if(self.Q_[d]==None or d in self.immutable_features_):
                    costs.append([ 0 ])
                else:
                    Q_d = self.Q_[d]
                    Q_0 = Q_d(x[d])
                    costs.append( [ abs(np.log2( (1-Q_d(x[d]+a)) / (1-Q_0) )) if cost_type=='TLPS' else abs( Q_d(x[d]+a) - Q_0) for a in self.actions_[d] ] )
        else:
            weights = self.getFeatureWeight(cost_type=cost_type)
            if(cost_type=='PCC'): p=2
            for d in range(self.D_):
                costs.append( list(weights[d] * abs(self.actions_[d])**p) )
            self.weights_ = weights
        return costs

    def getForestIntervals(self, x):
        Is = [np.arange(len(a)) for a in self.actions_]
        I = []
        for t in range(self.T_):
            I_t = []
            for l in range(self.L_[t]):
                I_t_l = []
                for d in range(self.D_):
                    xa = x[d] + self.actions_[d]
                    if isinstance(self.clf_, XGBClassifier):
                        I_t_l.append( list(((xa >= self.regions_[t][l][d][0]) & (xa < self.regions_[t][l][d][1])).astype(int)) )
                    else:
                        I_t_l.append( list(((xa > self.regions_[t][l][d][0]) & (xa <= self.regions_[t][l][d][1])).astype(int)) )
                I_t.append(I_t_l)
            I.append(I_t)
        return I

    def generateActions(self, X, cost_type='TLPS', p=1, union=False):
        if isinstance(self.clf_, XGBClassifier):
            for n in range(X.shape[0]):
                is_nan = np.isnan(X[0])
                X[n, is_nan] = self.X_ub_[is_nan]

        self.actions_ = self.getMultiActionSet(X, union=union)
        if(self.is_forest_): self.I_ =  [self.getForestIntervals(x) for x in X ]
        self.x_ = X
        self.costs_ = [ self.getCostSet(x, cost_type=cost_type, p=p) for x in X ]
        self.cost_type_ = cost_type
        return self.actions_, self.costs_

    def apply(self, x, a):
        x_cf = x + a
        for g in self.feature_categories_:
            if((x_cf[g]>=0).all() and (x_cf[g]<=1).all()): continue
            d_before = g[np.where(a[g]==-1)[0][0]]; d_after = g[np.where(x[g]==1)[0][0]]
            a[d_before] = 0; a[d_after] = -1;
        return a

    def counterfactual(self, x, a):
        return x + self.apply(x, a)

    def counterfactuals(self, X, a):
        return np.array([self.counterfactual(x, a) for x in X])

    def is_feasible(self, x, a):
        a = self.apply(x, a)
        for d in range(self.D_):
            if abs(a[d])<self.tol_: continue
            if(self.feature_types_[d]=='B'):
                if (int(x[d]+a[d]) not in [0, 1]): return False
            else:
                if (x[d]+a[d]<self.X_lb_[d] or x[d]+a[d]>self.X_ub_[d]): return False
        for g in self.feature_categories_:
            if((x[g]+a[g]).sum()!=1): return False
        return True

    def is_valid(self, x, a, y_target=0):
        return (self.clf_.predict( (self.counterfactual(x,a)).reshape(1,-1) )[0] == y_target) and self.is_feasible(x, a)

    def cost(self, x, a, cost_type='TLPS'):
        a = self.apply(x, a)
        cost = 0.0
        if(cost_type=='TLPS' or cost_type=='MPS'):
            if(self.Q_==None): 
                self.Q_ = [None if self.feature_constraints_[d]=='F' else CumulativeDistributionFunction(self.grids_[d], self.X_[:, d]) for d in range(self.D_)]
            for d in range(self.D_):
                if(self.Q_[d] is None): continue
                Q_d = self.Q_[d]; Q_0 = Q_d(x[d]);
                if(cost_type=='TLPS'):
                    cost += abs(np.log2( (1-Q_d(x[d]+a[d])) / (1-Q_0) ))
                else:
                    c = abs( Q_d(x[d]+a[d]) - Q_0 )
                    if(cost < c): cost = c
        else:
            if self.cost_type_==cost_type and self.weights_ is not None:
                weights = self.weights_
            else:
                weights = self.getFeatureWeight(cost_type=cost_type)
            p = 2 if cost_type=='PCC' else 1
            for d in range(self.D_): cost += weights[d] * (abs(a[d])**p)
        return cost

    # For RandomForestClassifier
    def getForestLabels(self):
        H = []
        for tree, leaves, l_t in zip(self.trees_, self.leaves_, self.L_):
            h_t=[]; stack=[ 0 ];
            while(len(stack)!=0):
                i = stack.pop()
                if(i in leaves):
                    if isinstance(self.clf_, XGBClassifier):
                        val = tree.value[i]
                        h_t += [ val ]
                    else:
                        val = tree.value[i][0]
                        h_t += [ val[0] if val.shape[0]==1 else val[1]/(val[0]+val[1]) ]
                else:
                    stack+=[ tree.children_right[i] ]; stack+=[ tree.children_left[i] ];
            H.append(h_t)
        return H

    # For RandomForestClassifier
    def getForestRegions(self):
        As, Rs = [], []
        for tree, leaves in zip(self.trees_, self.leaves_):
            A, R = [], []
            stack = [[]]
            L, U = [[-np.inf]*self.D_], [[np.inf]*self.D_]
            node_stack = [ 0 ]
            while(len(node_stack)!=0):
                n = node_stack.pop()
                a, l, u = stack.pop(), L.pop(), U.pop()
                if(n in leaves):
                    A.append(a)
                    R.append([ (l[d], u[d]) for d in range(self.D_)])
                else:
                    d = tree.feature[n]
                    if(d not in a): a_ = list(a) + [d]
                    stack.append(a_); stack.append(a_); 
                    b = tree.threshold[n]
                    l_ = list(l); u_ = list(u); 
                    l[d] = b; u[d] = b
                    U.append(u_); L.append(l); node_stack.append(tree.children_right[n]); 
                    U.append(u); L.append(l_); node_stack.append(tree.children_left[n]); 
            As.append(A); Rs.append(R)
        return As, Rs

    # For RandomForestClassifier
    def getForestThresholds(self):
        B = []
        for d in range(self.D_):
            b_d = []
            for tree in self.trees_: 
                b_d += list(tree.threshold[tree.feature==d])
            b_d = list(set(b_d))
            b_d.sort()
            B.append(np.array(b_d))
        return B

    # For RandomForestClassifier
    def getForestPartitions(self):
        I = []
        for t in range(self.T_):
            I_t = []
            for l in range(self.L_[t]):
                I_t_l = []
                for d in range(self.D_):
                    if(self.regions_[t][l][d][0]==-np.inf):
                        start = 0
                    else:
                        start = self.thresholds_[d].index(self.regions_[t][l][d][0]) + 1
                    if(self.regions_[t][l][d][1]== np.inf):
                        end = self.M_[d]
                    else:
                        end = self.thresholds_[d].index(self.regions_[t][l][d][1]) + 1
                    tmp = list(range(start, end))
                    I_t_l.append(tmp)
                I_t.append(I_t_l)
            I.append(I_t)
        return I

# class ActionCandidates



class Action():
    def __init__(self, x, a, scores={},
                 target_name='Output', target_labels=['Good', 'Bad'], label_before=1, label_after=0,
                 feature_names=[], feature_types=[], feature_categories=[], print_instance=False, print_features=[]):
        self.x_ = x
        self.a_ = a
        self.scores_ = scores
        self.target_name_ = target_name
        self.labels_ = [target_labels[label_before], target_labels[label_after]]
        self.feature_names_ = feature_names if len(feature_names)==len(x) else ['x_{}'.format(d) for d in range(len(x))]
        self.feature_types_ = feature_types if len(feature_types)==len(x) else ['C' for d in range(len(x))]
        self.feature_categories_ = feature_categories
        self.print_instance = print_instance
        self.print_features = print_features if len(print_features)>0 else list(range(len(x)))

        self.feature_categories_inv_ = []
        for d in range(len(x)):
            g = -1
            if(self.feature_types_[d]=='B'):
                for i, cat in enumerate(self.feature_categories_):
                    if(d in cat): 
                        g = i
                        break
            self.feature_categories_inv_.append(g)            

    def __str__(self):
        s = ''
        if(self.print_instance):
            s += '* Instance:\n' 
            for d in self.print_features:
                x_d = self.x_[d]
                g = self.feature_categories_inv_[d]
                if(g==-1):
                    if(self.feature_types_[d]=='C'):
                        s += '\t* {}: {:.4f}\n'.format(self.feature_names_[d], x_d) 
                    elif(self.feature_types_[d]=='B'):
                        s += '\t* {}: {}\n'.format(self.feature_names_[d], bool(x_d)) 
                    else:
                        s += '\t* {}: {}\n'.format(self.feature_names_[d], int(x_d))
                else:
                    if(x_d!=1): continue
                    s += '\t* {}\n'.format(self.feature_names_[d])
        s += '* Action ({}: {} -> {}):\n'.format(self.target_name_, self.labels_[0], self.labels_[1])
        i = 0
        for d in np.where(abs(self.a_)>1e-8)[0]:
            num = '*'
            g = self.feature_categories_inv_[d]
            if(g==-1):
                if(self.feature_types_[d]=='C'):
                    s += '\t{} {}: {:.4f} -> {:.4f} ({:+.4f})\n'.format(num, self.feature_names_[d], self.x_[d], self.x_[d]+self.a_[d], self.a_[d])
                elif(self.feature_types_[d]=='B'):
                    s += '\t* {}: True -> False\n'.format(self.feature_names_[d]) if bool(self.x_[d]) else '\t* {}: False -> True\n'.format(self.feature_names_[d])
                else:
                    s += '\t{} {}: {} -> {} ({:+})\n'.format(num, self.feature_names_[d], self.x_[d].astype(int), (self.x_[d]+self.a_[d]).astype(int), self.a_[d].astype(int))
            else:
                if(self.x_[d]==1): continue
                cat_name, nxt = self.feature_names_[d].split(':')
                cat = self.feature_categories_[g]
                prv = self.feature_names_[cat[np.where(self.x_[cat])[0][0]]].split(':')[1]
                s += '\t{} {}: {} -> {}\n'.format(num, cat_name, prv, nxt)

        if(len(self.scores_)>0):
            s += '* Scores: \n'
            for i in self.scores_.items():
                s += '\t* {0}: {1:.8f}\n'.format(i[0], i[1]) if isinstance(i[1], float) else '\t* {0}: {1}\n'.format(i[0], i[1])
        return s
    
# class Action

