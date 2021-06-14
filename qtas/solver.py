# -*- coding: utf-8 -*-
# © 2018-2019 Nokia
#
#Licensed under the BSD 3 Clause license
#SPDX-License-Identifier: BSD-3-Clause

from multiprocessing.pool import ThreadPool
# from copy import deepcopy
import numpy as np
# import datetime

from z3 import *  # @UnusedWildImport
from . import VeriGbError


class SolverManager():
    
    def __init__(self, matadata):
        self.FEATURES_VARS = {}
        self.AUX_VARS = {}
        self.OUTPUT_VAR = None
        self.PERTUB_VARS = {}
        self.PERTUB_PREFIX = "pertub_"
        
        #
        self.solver = Solver()
        #
        self.matadata = matadata
        # creating the actual variables 
        self.create_list_of_feature_var()
        self.create_output_var()
        
    def get_solver(self):
        return self.solver
    
    def set_solver_timeout(self, timeout):
        self.solver.set("timeout", timeout)
    
    def get_feature_names(self):
        return self.matadata.get_feature_names()
        
    ## VARIABLES
    def create_list_of_feature_var(self, var_type=Real):
        for name in self.get_feature_names():
            self.FEATURES_VARS[name] = var_type(name)
            #self.FEATURES_VARS[name] = var_type(name)
        return self.FEATURES_VARS
    
    def create_aux_var(self, name, var_type=Real):
        self.AUX_VARS[name] = var_type(name)
        #self.AUX_VARS[name] = var_type(name)
        return self.AUX_VARS[name]
        
    def create_output_var(self, name='OUTPUT', var_type=Real):
        self.OUTPUT_VAR = var_type(name)
        #self.OUTPUT_VAR = var_type(name)
        return self.OUTPUT_VAR
        
    def get_feature_var(self, name):
        return self.FEATURES_VARS[name]
    
    def get_list_of_feature_var(self):
        return self.FEATURES_VARS
    
    def get_aux_var(self, name):
        return self.AUX_VARS[name]

    def get_output_var(self):
        return self.OUTPUT_VAR
    

class SolverTask():
        
    def __init__(self, solver_mng, formal_model, prop, timeout=30000):
        self.solver_mng = solver_mng
        self.solver_mng.set_solver_timeout(timeout)
        self.formal_model = formal_model
        self.prop = prop
        #
        self.exe_solver = False
        self.exe_last_result = None
    
    def run(self):
        solver = self.solver_mng.get_solver()
        solver.add(self.formal_model)
        solver.add(Not(self.prop))
        
        check = solver.check()
        self.exe_solver = True
        
        if (check == z3.unknown):
            self.exe_last_result = "TimeOut"
        elif (check != z3.sat):
            self.exe_last_result = "Holds"
        else:
            self.exe_last_result = "Violated"
        return self.exe_last_result
    
    def has_counter_example(self):
        return self.exe_solver and (self.exe_last_result == "Violated")
    
    def get_last_result(self):
        if (not self.exe_solver):
            raise VeriGbError("Cannot extract a counter example without an exe command")            
        return self.exe_last_result
    
    def get_counter_example(self):
        if (not self.exe_solver):
            raise VeriGbError("Cannot extract a counter example without an exe command")            
                    
        if (self.exe_last_result != "Violated"): return (None, None)
        
        solver = self.solver_mng.get_solver()
        sol = solver.model()
        feature_names = self.solver_mng.get_feature_names()
        ret = np.zeros(len(feature_names))
        for idx, fe in enumerate(feature_names):
            val = sol[self.solver_mng.get_feature_var(fe)]
            if (val == None): val = 0
            elif (val.is_real()): val = val.as_fraction()
            elif (val.is_int()): val = val.as_string()
            ret[idx] = val
        
        gen_lab = sol[self.solver_mng.get_output_var()]
        if (gen_lab == None): gen_lab = -1
        elif (gen_lab.is_real()): gen_lab = gen_lab.as_fraction()
        elif (gen_lab.is_int()): gen_lab = gen_lab.as_string()
        
        return (gen_lab, ret)
    
    def get_serial_result(self):
        t_res = SolverTaskResult()
        #
        (gen_label, gen_obs) = (None, None)
        if (self.has_counter_example()):
            (gen_label, gen_obs) = self.get_counter_example()
        t_res.set_run_res(self.get_last_result())
        t_res.set_gen_label(gen_label)
        t_res.set_gen_obs(gen_obs)
        
        return t_res


# serializable result class (for pickling multiprocessing results)
class SolverTaskResult():
    def __init__(self, run_res=None, gen_label=None, gen_obs=None):
        self.run_res = run_res
        self.gen_label = gen_label
        self.gen_obs = gen_obs

    def get_run_res(self):
        return self.run_res

    def get_gen_label(self):
        return self.gen_label

    def get_gen_obs(self):
        return self.gen_obs

    def set_run_res(self, value):
        self.run_res = value

    def set_gen_label(self, value):
        self.gen_label = value

    def set_gen_obs(self, value):
        self.gen_obs = value

    def has_counter_example(self):
        return self.run_res == "Violated"

THREAD_POOL_SIZE = 8
def run_all_tasks(all_tasks, pool_size=THREAD_POOL_SIZE):
    task_pool = ThreadPool(pool_size)
    for t in all_tasks:
        task_pool.apply_async(t.run, args=[])
        
    task_pool.close()
    task_pool.join()

    all_tasks_results = []
    for t in all_tasks:
        a_res = (t.get_last_result(), t.get_last_result())
        if (t.has_counter_example()): a_res = t.get_counter_example()
        all_tasks_results.append(a_res)

    return all_tasks_results


