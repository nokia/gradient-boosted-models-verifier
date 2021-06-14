# -*- coding: utf-8 -*-
# © 2018-2019 Nokia
#
#Licensed under the BSD 3 Clause license
#SPDX-License-Identifier: BSD-3-Clause


import numpy
# import math
from z3 import *  # @UnusedWildImport

from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.tree import _tree
from xgboost.sklearn import XGBClassifier

from .xgbparser import parse_trees
from . import VeriGbError
from tkinter.tix import Form

print("dasd")

class IDecisionTreeEnsembleFormalRegressorModel():
    '''
    An interface for decision tree ensemble regressor or a single classifier class.
    When class_label is not provided, this represent a single regression model.
    If class_label is given, then it represent a single class within a classification model.
    '''
    
    def __init__(self, solver_mng, metadata, class_label):
        '''
        metadata.get_num_labels() must be strictly greater than class_label.
        regressor is the special case where metadata.get_num_labels()=1 and class_label=0.
        
        :param solver_mng: variable manager
        :param metadata:
        :param class_label:
        '''
        self.solver_mng = solver_mng
        self.class_label = class_label
        self.metadata = metadata
        if (self.class_label >= self.metadata.get_num_labels()):
            raise VeriGbError("wrong label invocation of tree ensemble")
        #
    
    def _close_construction(self,model_id):
        '''
        Can and must be called at the end of the child construction
        (only after the child information is all set up and ready...). 
        '''
        model_id=str(model_id)
        for tree_idx in range(self.get_num_trees()):
            self.solver_mng.create_aux_var("%sL%sOUT%s" % (str(model_id),self.get_label(), tree_idx))
        self.solver_mng.create_aux_var('L%sSUM%s' % (self.get_label(),model_id))
        self.solver_mng.create_output_var('OUTPUT%s' %model_id )

    def __get_sum_variable(self,i):
        new_sum='L%sSUM'+str(i)
        return self.solver_mng.get_aux_var(new_sum % self.get_label())
    
    def __get_tree_val_variable(self, tree_idx,i):
        return self.solver_mng.get_aux_var("%sL%sOUT%s" % (str(i),self.get_label(), tree_idx))
    
    def get_gt_constraints(self, other_regressor_ensemble):
        return (self.__get_sum_variable(1) > other_regressor_ensemble.__get_sum_variable(1))

    def get_label(self):
        return self.class_label 
    
    def get_num_classes(self):
        return self.metadata.get_num_labels()
    
    def get_ge_zero(self,i):
        return (self.__get_sum_variable(i) >= 0)

    def get_ensemble_expr(self, observation=None, epsilon=float('inf'), min_val=float('-inf'), max_val=float('inf'), optimize_l1=False, i=0):
        '''
        Note that in order to get a general ensemble expression (not related to an observation),
        you can simply invoke this function without setting anything to epsilon...
        In such a case observation will be ignored...
        
        Also note that the min_val, max_val (and epsilon as well) limits the tree walk (removes branches),
        it DOES NOT mean that the assigned value cannot be below or above resp.
        The restriction to the assignment needs to be done through the property!  

        :param observation:
        :param epsilon:
        :param min_val: 
        :param max_val:
        :param optimize_l1:
        '''
        trees_expr = True
        # TODO: check for z3.Sum
        sum_expr = self.get_base_predictor()
        for tree_idx in range(0, self.get_num_trees()):
            # evaluated the tree itself
            trees_expr = And(trees_expr, self.get_tree_clause_expr(tree_idx, observation, epsilon, min_val, max_val, optimize_l1,i))
            # accumulate the sum expression
            sum_expr = sum_expr + self.__get_tree_val_variable(tree_idx,i)
        # final expression
        return And(trees_expr, (self.__get_sum_variable(i) == sum_expr))
        
    def get_tree_clause_expr(self, tree_idx, observation=None, epsilon=float('inf'), min_val=float('-inf'), max_val=float('inf'), optimize_l1=False,indx=1):

        num_of_features = self.metadata.get_num_features()
        if (isinstance(min_val, list) and (len(min_val) != num_of_features)):
            raise VeriGbError("array length do not match")
        if (isinstance(max_val, list) and (len(max_val) != num_of_features)):
            raise VeriGbError("array length do not match")
        if (isinstance(epsilon, list) and (len(epsilon) != num_of_features)):
            raise VeriGbError("array length do not match")
        if (isinstance(epsilon, list) and optimize_l1):
            raise VeriGbError("epsilon cannot be a list for l1 optimisation")
        if (isinstance(observation, list) and (len(observation) != num_of_features)):
            raise VeriGbError("array length do not match")
        if (epsilon == float('inf')):
            observation = numpy.zeros(num_of_features)

        lower_range = numpy.zeros(num_of_features)
        upper_range = numpy.zeros(num_of_features)
        for i in range(num_of_features):
            min_v = min_val if (not isinstance(min_val, list)) else min_val[i] 
            max_v = max_val if (not isinstance(max_val, list)) else max_val[i]
            epsilon_v = epsilon if (not isinstance(epsilon, list)) else epsilon[i]
            #
            upper_range[i] = min(observation[i] + abs(epsilon_v), max_v)
            if (math.isinf(epsilon_v) and math.isinf(max_v)): upper_range[i] = float('inf')
            lower_range[i] = max(observation[i] - abs(epsilon_v), min_v)
            if (math.isinf(epsilon_v) and math.isinf(min_v)): lower_range[i] = float('-inf')
        
        tree_instance = self.get_tree(tree_idx)
        
        def build_tree_clause_helper(node_idx, depth, ret_constraint_list, accum_min_epsi, rec_accum=[],indx=1):
            feature_idx = tree_instance.feature[node_idx]
            # stop case...
            if(self.is_leaf(tree_idx, node_idx)):
                # val = self.get_learning_rate() * self.get_node_value(tree_idx, node_idx)
                val = self.get_node_value(tree_idx, node_idx)
                rec_accum.append(self.__get_tree_val_variable(tree_idx,indx) == val)
                if (not optimize_l1): 
                    ret_constraint_list.append(And(*rec_accum))
                # else, optimize_l1 is True
                elif ((not isinstance(epsilon, list)) and (accum_min_epsi <= epsilon)):
                    ret_constraint_list.append(And(*rec_accum))
                # else: # not adding this path 
                #     print("-- removing un-necessary branch...!")
                rec_accum.pop()
                return

            # else..
            
            FEATURE_VAR = self.solver_mng.get_feature_var(self.metadata.get_feature_names(feature_idx))
            #==============================================================
            # Note: if the lower range is not relevant, then it will skip this part. 
            new_accum_min_epsi = accum_min_epsi
            if (self.get_node_condition(tree_idx, node_idx, lower_range[feature_idx])):
                if (isinstance(observation, list) and (not self.get_node_condition(tree_idx, node_idx, observation[feature_idx]))):
                    new_accum_min_epsi += abs(observation[feature_idx] - self.get_node_threshold(tree_idx, node_idx))
                rec_accum.append(self.get_node_condition(tree_idx, node_idx, FEATURE_VAR))
                build_tree_clause_helper(self.get_true_child(tree_idx, node_idx), depth + 1, ret_constraint_list, new_accum_min_epsi, rec_accum,indx)
                rec_accum.pop()
            
            new_accum_min_epsi = accum_min_epsi
            if (not self.get_node_condition(tree_idx, node_idx, upper_range[feature_idx])):
                if (isinstance(observation, list) and (self.get_node_condition(tree_idx, node_idx, observation[feature_idx]))):
                    new_accum_min_epsi += abs(self.get_node_threshold(tree_idx, node_idx) - observation[feature_idx]) 
                rec_accum.append(Not(self.get_node_condition(tree_idx, node_idx, FEATURE_VAR)))
                build_tree_clause_helper(self.get_false_child(tree_idx, node_idx), depth + 1, ret_constraint_list, new_accum_min_epsi, rec_accum,indx)
                rec_accum.pop()
        
        # # invoking the recursive all...
        constraint_list = []
        build_tree_clause_helper(0, 1, constraint_list, 0,indx=indx)
        return Or(*constraint_list)
    
    #===========================================================================
    # # abstract interfaces
    #===========================================================================
    def get_base_predictor(self):
        raise NotImplementedError("Not implemented yet")
    
    def get_learning_rate(self):
        raise NotImplementedError("Not implemented yet")

    def get_num_trees(self):
        raise NotImplementedError("Not implemented yet")
    
    def get_tree(self, tree_idx):
        raise NotImplementedError("Not implemented yet")        

    def get_node_condition(self, tree_idx, node_idx, val):
        '''
        Note that this getter can be used with a scalar value 'val', in which case
        it will return True/False, or an SMT variable in which case it will return
        an expression.
        :param tree_idx:
        :param node_idx:
        :param val:
        '''
        raise NotImplementedError("Not implemented yet")
    
    def get_true_child(self, tree_idx, node_idx):
        raise NotImplementedError("Not implemented yet")
    
    def get_false_child(self, tree_idx, node_idx):
        raise NotImplementedError("Not implemented yet")

    def is_leaf(self, tree_idx, node_idx):
        raise NotImplementedError("Not implemented yet")

    def get_node_value(self, tree_idx, node_idx):
        raise NotImplementedError("Not implemented yet")

    def get_node_threshold(self, tree_idx, node_idx):
        raise NotImplementedError("Not implemented yet")


# Adapter  
class SklFormalRegressorModel(IDecisionTreeEnsembleFormalRegressorModel):
    '''
    sklearn implementation
    '''
    
    def __init__(self, solver_mng, metadata, skl_gb_regression_model, class_label,model_id):
        IDecisionTreeEnsembleFormalRegressorModel.__init__(self, solver_mng, metadata, class_label)
        self.model = skl_gb_regression_model;
        # only now we know the amount of trees --> finishing the super constructor
        IDecisionTreeEnsembleFormalRegressorModel._close_construction(self, model_id)

    def get_base_predictor(self):
        # FIXME: check if correct for regressor case 
        return self.model.init_.priors[self.get_label()]

    def get_learning_rate(self):
        return self.model.learning_rate
    
    def get_num_trees(self):
        return len(self.model.estimators_)  # actually instantiated
    
    def get_tree(self, tree_idx):
        # FIXME: check if correct for regressor case 
        return self.model.estimators_[tree_idx, self.get_label()].tree_
    
    #===========================================================================
    # Note that the condition is different than xgboost... 
    #===========================================================================
    def get_node_condition(self, tree_idx, node_idx, val):
        if (isinstance(val , float) and (val == float("-inf"))): return True
        if (isinstance(val , float) and (val == float("inf"))): return False
        return (val <= self.get_node_threshold(tree_idx, node_idx))
    
    def get_true_child(self, tree_idx, node_idx):
        return self.get_tree(tree_idx).children_left[node_idx]
    
    def get_false_child(self, tree_idx, node_idx):
        return self.get_tree(tree_idx).children_right[node_idx]

    def is_leaf(self, tree_idx, node_idx):
        a_tree = self.get_tree(tree_idx)
        feature_idx = a_tree.feature[node_idx]
        if (feature_idx == None): return True
        # else
        #
        # old version works but doesn't parse well...
        # return (feature_idx == _tree.TREE_UNDEFINED)
        return (a_tree.children_left[node_idx] == a_tree.children_right[node_idx]) 

    def get_node_value(self, tree_idx, node_idx):
        return self.get_tree(tree_idx).value[node_idx][0][0]

    def get_node_threshold(self, tree_idx, node_idx):
        return self.get_tree(tree_idx).threshold[node_idx]


# 
class XgbFormalRegressorModel(IDecisionTreeEnsembleFormalRegressorModel):
    '''
    xgboost implementation
    '''
    
    def __init__(self, solver_mng, metadata, xgb_regression_model, class_label,model_id):
        IDecisionTreeEnsembleFormalRegressorModel.__init__(self, solver_mng, metadata, class_label)
        self.model = xgb_regression_model;
        self.trees = self._extract_trees_workaround(parse_trees(self.model))
        # only now we know the amount of trees --> finishing the super constructor
        IDecisionTreeEnsembleFormalRegressorModel._close_construction(self,model_id)
    
    def _extract_trees_workaround(self, all_trees):
        ret_trees = []
        for tree_idx in range(len(all_trees)):
            # FIXME: not sure it will work well for regressor
            if ((tree_idx % self.get_num_classes()) == self.get_label()):
                ret_trees.append(all_trees[tree_idx])
        return ret_trees

    def get_base_predictor(self):
        return 0

    def get_learning_rate(self):
        # not really relevant... multiplying by 1
        return 1  # self.model.learning_rate
    
    def get_num_trees(self):
        return len(self.trees)
    
    def get_tree(self, tree_idx):
        return self.trees[tree_idx]
    
    #===========================================================================
    # Note that the condition is different than sklearn... 
    #===========================================================================
    def get_node_condition(self, tree_idx, node_idx, val):
        if (isinstance(val , float) and (val == float("-inf"))): return True
        if (isinstance(val , float) and (val == float("inf"))): return False
        return (val < self.get_node_threshold(tree_idx, node_idx))

    def get_true_child(self, tree_idx, node_idx):
        return self.get_tree(tree_idx).true_children[node_idx]
    
    def get_false_child(self, tree_idx, node_idx):
        return self.get_tree(tree_idx).false_children[node_idx]

    def is_leaf(self, tree_idx, node_idx):
        feature_idx = self.get_tree(tree_idx).feature[node_idx]
        return (feature_idx == None)

    def get_node_value(self, tree_idx, node_idx):
        # FIXME: bug! not the threshold!!
        return self.get_tree(tree_idx).threshold[node_idx]

    def get_node_threshold(self, tree_idx, node_idx):
        return self.get_tree(tree_idx).threshold[node_idx]


#==============================================================================
# ensembles interfaces
#==============================================================================
def gb_classification(solver_mng, metadata, model, model_id=1,cmp_label=None, obs_vec=None, obs_label=None, epsilon=float('inf'), min_val=float('-inf'), max_val=float('inf')):
    #print("building gb_classification...")
    formal_model = []
    for class_label in range(metadata.get_num_labels()):
        if (isinstance(model, GradientBoostingClassifier)):
            formal_model.append(SklFormalRegressorModel(solver_mng, metadata, model, class_label,model_id))
        elif(isinstance(model, XGBClassifier)):
            formal_model.append(XgbFormalRegressorModel(solver_mng, metadata, model, class_label,model_id))
        else:
            raise VeriGbError("unknown model type '" + str(type(model)) + "'")
    
    if (cmp_label == None):
        cmp_label = []
        for i in range(metadata.get_num_labels()): cmp_label.append(i)
    
    # cmp must have at least obs_label
    if (obs_label != None): cmp_label.append(obs_label)
    elif (epsilon != float('inf')):
        raise VeriGbError("epsilon cannot be set if an observation is not given")
    
    if (len(cmp_label) <= 1):
        raise VeriGbError("Must have at least 2 labels to compare between")
    print(formal_model)
    class_alt = False
    OUTPUT_VAR = solver_mng.get_output_var()
    for l1 in cmp_label: class_alt = Or(class_alt, (OUTPUT_VAR == l1))
    # 
    mc_model = And(True, class_alt)
    for l1 in cmp_label:
        mc_model = And(mc_model, formal_model[l1].get_ensemble_expr(obs_vec, epsilon, min_val, max_val, optimize_l1=False,i=1))
        l1_gt_others = True
        for l2 in cmp_label:
            if (l1 == l2): continue
            l1_gt_others = And(l1_gt_others, formal_model[l1].get_gt_constraints(formal_model[l2]))
        mc_model = And(mc_model, ((OUTPUT_VAR == l1) == l1_gt_others))
        
    # print("======================== mc_model:\n" + str(mc_model))
    return mc_model


def gb_binary_classification(solver_mng, metadata, model, obs_vec, obs_label,model_id=1 ,
                        epsilon=float('inf'), min_val=float('-inf'), max_val=float('inf')):
    #print("building gb_binary_classification...")
    #obs label is the original
    #cmp_lable label to compare to
    formal_model = None
    if (isinstance(model, GradientBoostingClassifier)):
        formal_model = SklFormalRegressorModel(solver_mng, metadata, model, 0)
    elif(isinstance(model, XGBClassifier)):
        formal_model = XgbFormalRegressorModel(solver_mng, metadata, model, 0,model_id)
    else:
        raise VeriGbError("unknown model type '" + str(type(model)) + "'")
    
  
    #mc_model = formal_model.get_ensemble_expr(obs_vec, epsilon, min_val, max_val, optimize_l1=False)
    #OUTPUT_VAR = solver_mng.get_output_var()
    
    #mc_model = And(mc_model, ((OUTPUT_VAR == 1) == formal_model.get_ge_zero()))
    
    #mc_model = And(mc_model, ((OUTPUT_VAR == 0) == Not(formal_model.get_ge_zero())))
    
    return formal_model

    
def rf_classification():
    pass
    
def rf_unsupervised():
    pass
    
#==============================================================================
# def iff_z3(a, b):
#    return And(Implies(a, b), Implies(b, a))

