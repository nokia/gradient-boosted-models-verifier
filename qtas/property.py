# -*- coding: utf-8 -*-
# © 2018-2019 Nokia
#
#Licensed under the BSD 3 Clause license
#SPDX-License-Identifier: BSD-3-Clause


from . import VeriGbError

from z3 import *  # @UnusedWildImport

# TODO: think about a general type checking!!
# see DTYPE_MAPPER's in xgboost->python-package->xgboost->core.py


def abs_z3(x):
    return If(x >= 0, x, -x)


def max_z3(x, y):
    return If(x >= y, x, y)


#def cmp_z3(a, b):
#    return If(a <= b, 0, 1)


def robustness_L1_property(solver_mng, metadata, obs_vec, obs_label, epsilon, min_val=float('-inf'), max_val=float('inf'), delta=None):
    '''
    Construct the robustness, for L1 norm, i.e the sum of distances is less than epsilon.
    
    Note that for L1 we can support only Real features since the epsilon restriction is
    only applied over the summation of all features.
    
    delta can either be None (no change to label), value (bound for the change in the label
    -- relevant of regression only), or list of value (that are allowed in addition to label)
    list of values can be used to either support Int output (in which in between values are
    not allowed), or for classification in which we are not interested in counter examples
    for specific outputs (that are in the delta). 
    
    Note that for the list cases, epsilon is the allowed change with respect to the input,
    while in delta case is the values themselves (not related to the label).
    * for future reference, we might consider to consolidate it somehow to behave similarly 
    (this will also enable support for categorical features noted above).

    :param solver_mng: variables manager 
    :param metadata: 
    :param obs_vec: list of input values (an observation vector)
    :param obs_label: expected output (the observation label)
    :param epsilon: allowed perturbation.  
    :param min_val: minimum value.  
    :param man_val: maximum value.  
    :param delta: allowed output change
    '''
    
    num_of_features = metadata.get_num_features()

    # CHECK PARAMS (can be extended...)
    if (isinstance(epsilon, list)): raise VeriGbError("epsilon cannot be a list for L1 distance")
    if (num_of_features != len(obs_vec)): raise VeriGbError("observation and FEATURES must be of the same length")
    
    # NOTE: Real/Int values 
    assume = True
    diff_sum = 0 
    for i in range(num_of_features):
        FEATURE_VAR = solver_mng.get_feature_var(metadata.get_feature_names(i))
        #PERTUB_VAR = solver_mng.get_pertub_var(metadata.get_feature_names(i))
        #assume = And(assume, (FEATURE_VAR - obs_vec[i]) == PERTUB_VAR)
        #
        diff_sum = diff_sum + abs_z3(FEATURE_VAR - obs_vec[i])
        if (min_val != float("-inf")): assume = And(assume, FEATURE_VAR >= min_val)
        if (max_val != float("inf")): assume = And(assume, FEATURE_VAR <= max_val)
    assume = And(assume, (diff_sum <= epsilon))
    
    conseq = False
    OUTPUT_VAR = solver_mng.get_output_var()
    # TODO: re-think the following int() casting? (should it maybe be float()?)
    if (delta == None):
        # check for any change in output
        conseq = Or(conseq, OUTPUT_VAR == obs_label)
    elif (isinstance(delta, list)):
        conseq = Or(conseq, (OUTPUT_VAR == obs_label))  # either the label
        # or OUTPUT_VAR can be set to several values (others are counter examples) 
        for j in range(0, len(delta)):  # and in the delta
            conseq = Or(conseq, (OUTPUT_VAR == int(delta[j])))
    else:
        # delta is a bound... 
        conseq = Or(conseq, (abs_z3(OUTPUT_VAR - obs_label) <= delta))
    
    # property...
    return Implies(assume, conseq)

    
def robustness_Linf_property(solver_mng, metadata, obs_vec, obs_label, epsilon, min_val=float('-inf'), max_val=float('inf'), delta=None):
    '''
    Construct the robustness, for L inf norm, i.e each distances vector is less than epsilon.
    
    epsilon can be given a:
    - single "epsilon value" to all FEATURES_VARS, or
    - list of "epsilon values" per FEATURES_VARS 
    Where "epsilon value" can be either a scalar value of the bound, or a change-set that lists
    all allowed perturbation. Change-set enables support Int features in which values in between
    two consecutive values, are not allowed. 
    
    e.g. FEATURE_VARS with 4 elements, can be coupled with:
    epsilon = [0.2, [0.05, 0.1, 0.15, 0.2], 0.2, 0]
    The first and third elements are bounds of upto 0.2 perturbation (for Real features)
    The second element allows a change of 4 specific values [0.05, 0.1, 0.15, 0.2] (for Int features)
    The last element is not allowed to be changed.
    * any way to support categorical feature?
    
    delta can either be None (no change to label), value (bound for the change in the label
    -- relevant of regression only), or list of value (that are allowed in addition to label)
    list of values can be used to either support Int output (in which in between values are
    not allowed), or for classification in which we are not interested in counter examples for
    specific outputs (that are in the delta). 
    
    Note that for the list cases, epsilon is the allowed change with respect to the input,
    while in delta case is the values themselves (not related to the label).
    * for future reference, we might consider to consolidate it somehow to behave similarly 
    (this will also enable support for categorical features noted above).
    
      
    :param solver_mng: variables manager 
    :param metadata: 
    :param obs_vec: list of input values (an observation vector)
    :param obs_label: expected output (the observation label)
    :param epsilon: allowed perturbation.  
    :param min_val: minimum value.  
    :param man_val: maximum value.  
    :param delta: allowed output change
    '''
    
    num_of_features = metadata.get_num_features()

    # Allow epsilon to be either: (i) array assigning epsilon per FEATURES_VARS, or (ii) a single
    # value that generates an array of such length. 
    if (isinstance(epsilon, list) and (num_of_features != len(epsilon))):
        raise VeriGbError("epsilon list must be with the same length of FEATURES")
    epsilon_list = epsilon if isinstance(epsilon, list) else [epsilon] * num_of_features 
    if (num_of_features != len(obs_vec)):
        raise VeriGbError("observation and FEATURES must be of the same length")

    # TODO: implement non-relative epsilon
    # for i in range(0, len(FEATURES_VARS)):
    #    if (isinstance(epsilon_list[i], list)):
    #        for j in range(0, len(epsilon_list[i])):
    #            # build here the non-relative epsilon to be similar to the otuput case 
    #            # assume = Or(assume, (abs_z3(FEATURES_VARS[i] - observation[i]) == epsilon_list[i][j]))

    assume = True
    # input
    for i in range(num_of_features):
        FEATURE_VAR = solver_mng.get_feature_var(metadata.get_feature_names(i))
        #PERTUB_VAR = solver_mng.get_pertub_var(metadata.get_feature_names(i))
        #assume = And(assume, (FEATURE_VAR - obs_vec[i]) == PERTUB_VAR)
         
        if (isinstance(epsilon_list[i], list)):
            # the case of set of values (categorical feature values)
            assume_val = (FEATURE_VAR == obs_vec[i])  # either is equal -- 0
            for j in range(0, len(epsilon_list[i])):
                assume_val = Or(assume_val, (abs_z3(FEATURE_VAR - obs_vec[i]) == epsilon_list[i][j]))
            assume = And(assume, assume_val)
        else:
            # the case of a bound value (Real feature)
            assume = And(assume, abs_z3(FEATURE_VAR - obs_vec[i]) <= epsilon_list[i])
        
        if (min_val != float("-inf")): assume = And(assume, FEATURE_VAR >= min_val)
        if (max_val != float("inf")): assume = And(assume, FEATURE_VAR <= max_val)
            
            
    conseq = False
    OUTPUT_VAR = solver_mng.get_output_var()
    # TODO: re-think the following int() casting? (should it maybe be float()?)
    if (delta == None):
        # check for any change in output
        conseq = Or(conseq, OUTPUT_VAR == obs_label)
    elif (isinstance(delta, list)):
        conseq = Or(conseq, (OUTPUT_VAR == obs_label))  # either the label
        # or OUTPUT_VAR can be set to several values (others are counter examples) 
        for j in range(0, len(delta)):  # and in the delta
            conseq = Or(conseq, (OUTPUT_VAR == int(delta[j])))
    else:
        # delta is a bound... 
        conseq = Or(conseq, (abs_z3(OUTPUT_VAR - obs_label) <= delta))
    
    # property...
    return Implies(assume, conseq)


def monotone_property(FEATURES_VARS,
                      OUTPUT_VAR,
                      observation,
                      label,
                      epsilon,
                      delta=None):
    '''
    TODO: not implemented yet!
    
    :param FEATURES_VARS:
    :param OUTPUT_VAR:
    :param observation:
    :param label:
    :param epsilon:
    :param delta:
    '''
    raise VeriGbError("Monotone property is not implemented yet!")

# def verify_monotonicity(self,
#                         mlmodel,
#                         epsi,
#                         timeout=60000,
#                         running_process=0) :
#     self.solver.set("timeout", timeout)
#     
#     for i in range(0, len(self.feature_names)):
#         f_1 = self.feature_names[i] + "_1"
#         f_2 = self.feature_names[i] + "_2"
#         self.solver.add(self.FEATURES[f_2] - self.FEATURES[f_1] <= epsi)
#         self.solver.add(self.FEATURES[f_2] - self.FEATURES[f_1] >= 0) 
#     
#     self.solver.add(self.CLASSIFY_1 == self.classYes)  # No Error
#     self.solver.add(self.CLASSIFY_2 == self.classNo)  # Error
#     
#     sum_expr_1 = 0
#     sum_expr_2 = 0
#     tree_id = 0
#     if isinstance(mlmodel, DecisionTreeClassifier):
#         self.robust_analysis_constraints(mlmodel.tree_, tree_id, None, "_1", epsi)
#         self.robust_analysis_constraints(mlmodel.tree_, tree_id, None, "_2", epsi)
#         self.solver.add(self.CLASSIFY_1 == self.OUT[str(tree_id) + "_1"])  # No Error
#         self.solver.add(self.CLASSIFY_2 == self.OUT[str(tree_id) + "_2"])  # Error
#     else:
#         for est in mlmodel.estimators_:
#         # while (tree_id < 4):
#          #   est = mlmodel.estimators_[tree_id]
#             
#             # self.dumpToPdf(est, tree_id)
#             
#             self.robust_analysis_constraints(est.tree_, tree_id, None, "_1", epsi)
#             self.robust_analysis_constraints(est.tree_, tree_id, None, "_2", epsi)
#             sum_expr_1 = sum_expr_1 + self.OUT[str(tree_id) + "_1"]
#             sum_expr_2 = sum_expr_2 + self.OUT[str(tree_id) + "_2"]
#             tree_id = tree_id + 1
#         
#         half_trees = 68
#         # half_trees = self.trees_n / 2
#         # half_trees = 2
#         print (half_trees)
#         self.solver.add(self.CLASSIFY_1 == cmp_z3(sum_expr_1, half_trees))  # Error
#         self.solver.add(self.CLASSIFY_2 == cmp_z3(sum_expr_2, half_trees))  # No Error
#         
#     with open('../logs/fcc/constraints.txt', 'w') as f:
#         f.write(str(self.solver))  
# 
#     logging.info("[%s] - checking monotonicity... %0.2fs" % (self.test_name, epsi))
#     
#     time_0 = time.time()
#     gen_sol = self.solve()
#     time_solve = time.time()
#     logging.info("[%s] - solving time: %0.2fs" % (self.test_name, (time_solve - time_0)))
#     print(str(gen_sol))
#    
#     self.post_process_result(gen_sol)


def one_pert_fits_all_Linf_property(solver_mng, metadata, obs_vecs, obs_labels, epsilon, delta=None):
    
    num_of_features = metadata.get_num_features()

    # Allow epsilon to be either: (i) array assigning epsilon per FEATURES_VARS, or (ii) a single
    # value that generates an array of such length. 
    if (isinstance(epsilon, list) and (num_of_features != len(epsilon))):
        raise VeriGbError("epsilon list must be with the same length of FEATURES")
    epsilon_list = epsilon if isinstance(epsilon, list) else [epsilon] * num_of_features 
    if (num_of_features != len(obs_vecs[0])):
        raise VeriGbError("observation and FEATURES must be of the same length")

    # TODO: implement non-relative epsilon
    # for i in range(0, len(FEATURES_VARS)):
    #    if (isinstance(epsilon_list[i], list)):
    #        for j in range(0, len(epsilon_list[i])):
    #            # build here the non-relative epsilon to be similar to the otuput case 
    #            # assume = Or(assume, (abs_z3(FEATURES_VARS[i] - observation[i]) == epsilon_list[i][j]))

    assume = True
    # input
    for j in range(len(obs_vecs)):
        obs_vec = obs_vecs[j]
        obs_label = obs_labels[j]
        
        for i in range(num_of_features):
            FEATURE_VAR = solver_mng.get_feature_var(metadata.get_feature_names(i))
            PERTUB_VAR = solver_mng.get_pertub_var(metadata.get_pertub_names(i))
            assume = And(assume, (FEATURE_VAR - obs_vec[i]) == PERTUB_VAR)
            
            if (isinstance(epsilon_list[i], list)):
                # the case of set of values (categorical feature values)
                assume_val = (FEATURE_VAR == obs_vec[i])  # either is equal -- 0
                for j in range(0, len(epsilon_list[i])):
                    assume_val = Or(assume_val, (abs_z3(FEATURE_VAR - obs_vec[i]) == epsilon_list[i][j]))
                assume = And(assume, assume_val)
            else:
                # the case of a bound value (Real feature)
                assume = And(assume, abs_z3(FEATURE_VAR - obs_vec[i]) <= epsilon_list[i])
            
                
        conseq = False
        OUTPUT_VAR = solver_mng.get_output_var()
        # TODO: re-think the following int() casting? (should it maybe be float()?)
        if (delta == None):
            # check for any change in output
            conseq = Or(conseq, OUTPUT_VAR == obs_label)
        elif (isinstance(delta, list)):
            conseq = Or(conseq, (OUTPUT_VAR == obs_label))  # either the label
            # or OUTPUT_VAR can be set to several values (others are counter examples) 
            for j in range(0, len(delta)):  # and in the delta
                conseq = Or(conseq, (OUTPUT_VAR == int(delta[j])))
        else:
            # delta is a bound... 
            conseq = Or(conseq, (abs_z3(OUTPUT_VAR - obs_label) <= delta))
        
    
    # property...
    return Implies(assume, conseq)
