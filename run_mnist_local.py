# -*- coding: utf-8 -*-
# © 2018-2019 Nokia
#
#Licensed under the BSD 3 Clause license
#SPDX-License-Identifier: BSD-3-Clause

from qtas.ensemble import gb_classification
from qtas.solver import SolverManager, SolverTask, run_all_tasks
from qtas.metadata import Metadata

# visualize in xgboost:
# https://machinelearningmastery.com/visualize-gradient-boosting-decision-trees-xgboost-python/

import warnings
from qtas.property import robustness_L1_property
from util import my_fetch_mldata, train_classifier_model,\
    export_image_example_to_pdf
warnings.filterwarnings(action='ignore', category=DeprecationWarning)


def main():
    myseed = 42
    train_model = False
   # train_model = True
    model_type = 'xgb'
   # model_type = 'skl'
    pkl_file = './resources/pkl/test-' + model_type + '-mnist.pkl'
    shuffle = True
    #
    data_home = None
    load_ratio = 1.0
    image_idx = 0
    image_min_pixel_val = 0
    image_max_pixel_val = 255
    image_flip = True # change to 
    image_scheme = 1
    image_cmap = 'gray'
    #
    # implementation specific
    # print_diff = False
    print_diff = True
    
    ######################################################################
    # 1. working on the data...
    print("#1: Loading data")
  #  raw = my_fetch_mldata('MNIST original', data_home=data_home, load_ratio=load_ratio)
    raw = my_fetch_mldata('mnist_784', data_home=data_home, load_ratio=load_ratio)
    mdata = Metadata.raw_data_constructor(data=raw.data, labels=raw.target, shuffle=shuffle, test_size=0.25, seed=myseed)
    mdata.set_image_feature_names(image_scheme=image_scheme)
    print("- Got %i training samples and %i test samples." % (len(mdata.get_train_data()), len(mdata.get_test_data())))
    
    ######################################################################
    # 2. training the model
    print("#2: Training model")
    mymodel = train_classifier_model(metadata=mdata, model_file=pkl_file, model_type=model_type,
                                     train_model=train_model, learn_r=0.1, esti_n=50, maxd=3, seed=myseed)
    

    ######################################################################
    # 3. working on the data...
    print("#3: Creating a variable manager")
    solver_mng = SolverManager(mdata)    
    
    ######################################################################
    # 4. constructing formal model
    print("#4: Building formal model")
    # - model over label 9, and image label 
    obs_vec = mdata.get_test_data(image_idx)
    obs_label = int(mdata.get_test_label(image_idx))
    # formal_model = gb_classification(solver_mng, mdata, mymodel, cmp_label=[9], obs_vec=obs_vec, obs_label=obs_label, epsilon=5, min_val=image_min_pixel_val, max_val=image_max_pixel_val)
    # - model over label 4, 9, and image label
    # formal_model = gb_classification(solver_mng, mdata, mymodel, cmp_label=[9,4], obs_vec=obs_vec, obs_label=obs_label, epsilon=6, min_val=image_min_pixel_val, max_val=image_max_pixel_val)
    # - model over all labels
    # formal_model = gb_classification(solver_mng, mdata, mymodel, obs_vec=obs_vec, obs_label=obs_label, epsilon=1, min_val=image_min_pixel_val, max_val=image_max_pixel_val)
    # formal_model = gb_classification(solver_mng, mdata, mymodel, cmp_label=[0,1,2,3,4,6,8,9], obs_vec=obs_vec, obs_label=obs_label, epsilon=1, min_val=image_min_pixel_val, max_val=image_max_pixel_val)
    # formal_model = gb_classification(solver_mng, mdata, mymodel, cmp_label=[0,1,4,6,8,9], obs_vec=obs_vec, obs_label=obs_label, epsilon=2, min_val=image_min_pixel_val, max_val=image_max_pixel_val)
    # formal_model = gb_classification(solver_mng, mdata, mymodel, cmp_label=[0,1,4,8,9], obs_vec=obs_vec, obs_label=obs_label, epsilon=7, min_val=image_min_pixel_val, max_val=image_max_pixel_val)
    # formal_model = gb_classification(solver_mng, mdata, mymodel, cmp_label=[0,1,4,9], obs_vec=obs_vec, obs_label=obs_label, epsilon=95, min_val=image_min_pixel_val, max_val=image_max_pixel_val)
    # formal_model = gb_classification(solver_mng, mdata, mymodel, cmp_label=[0,1,9], obs_vec=obs_vec, obs_label=obs_label, epsilon=173, min_val=image_min_pixel_val, max_val=image_max_pixel_val)
    # formal_model = gb_classification(solver_mng, mdata, mymodel, cmp_label=[1,9], obs_vec=obs_vec, obs_label=obs_label, epsilon=218, min_val=image_min_pixel_val, max_val=image_max_pixel_val)
    # - model without observation and no epsilon in the model itself... (just in the property)
    # formal_model = gb_classification(solver_mng, mdata, mymodel, cmp_label=[0,1,4,8,9], min_val=image_min_pixel_val, max_val=image_max_pixel_val)
    # - model without observation, without bound on the range, and no epsilon in the model itself... (just in the property)
    # formal_model = gb_classification(solver_mng, mdata, mymodel, cmp_label=[0,1,4,8,9])
    # formal_model = gb_classification(solver_mng, mdata, mymodel, cmp_label=[4,7])
    # formal_model = gb_classification(solver_mng, mdata, mymodel)
    #
    formal_model_breakdown = []
    formal_model_breakdown.append(gb_classification(solver_mng, mdata, mymodel, min_val=image_min_pixel_val, max_val=image_max_pixel_val))
   
    ######################################################################
    # 5. building the relevant property
    print("#5: Constructing the property")
    #epsilon_list = [[0,5,6,7,8]] * mdata.get_num_features()
    # prop_expr = robustness_L1_property(solver_mng, mdata, obs_vec, obs_label, epsilon=6)
    # # items can be lists as well
    # epsilon_list[0] = [0.25, 5.75]
    # epsilon_list = [[0, 8]] * mdata.get_num_features() 
    # # Linf property can have a list of epsilon values
    #prop_expr = robustness_Linf_property(solver_mng, mdata, obs_vec, obs_label, epsilon=epsilon_list)
    
    # prop_expr = robustness_Linf_property(solver_mng, mdata, obs_vec, obs_label, epsilon=8, min_val=image_min_pixel_val, max_val=image_max_pixel_val)
    prop_expr = robustness_L1_property(solver_mng, mdata, obs_vec, obs_label, epsilon=10, min_val=image_min_pixel_val, max_val=image_max_pixel_val)
    

    ######################################################################
    # 6. invoking a solver task
    print("#6: Constructing the task")
    # tasks_results = build_and_run_tasks(solver_mng, mdata, formal_model, prop_expr, timeout=30000)
    all_tasks = []
    for a_formal_model in formal_model_breakdown:
        all_tasks.append(SolverTask(solver_mng, a_formal_model, prop_expr, timeout=30000))
    tasks_results = run_all_tasks(all_tasks)
    res_idx = 0
    for a_res in tasks_results:
        # each element is (gen_label, gen_vec)
        res_idx += 1
        (gen_label, gen_vec) = a_res
        #gen_vec =  a_res
        #gen_label = obs_label
        if (isinstance(gen_label, str)): ## non counter examples return strings...
            print("- No counter example idx-" + str(res_idx) + " output: " + str(gen_label));        
        else:
            src_vec = mdata.get_test_data(image_idx)
            print("- Counter example idx-" + str(res_idx) + " with value " + str(gen_label) + ", instead of label " + str(int(mdata.get_test_label(image_idx))));
            print("- Counter example idx-" + str(res_idx));
            print("- Original model predicted value for gen: " + str(int(mymodel.predict(gen_vec.reshape(1, -1)))) + "; and predicted for source: " + str(int(mymodel.predict(src_vec.reshape(1, -1)))));
            file_name = "./resources/ce/mnist-ce" + str(image_idx) + "-" + model_type + "-idx-" + str(res_idx) + ".pdf"
            export_image_example_to_pdf(file_name=file_name,
                                        src_image_vec=src_vec,
                                        gen_image_vec=gen_vec,
                                        image_height=mdata.get_image_height(),
                                        image_width=mdata.get_image_width(),
                                        image_scheme=image_scheme,
                                        image_flip=image_flip,
                                        min_val=image_min_pixel_val,
                                        max_val=image_max_pixel_val,
                                        cmap=image_cmap)
            
            num_of_features = mdata.get_num_features()
            if (print_diff):
                print("print changes for idx-" + str(res_idx))
                for i in range(num_of_features):
                    diff = src_vec[i] - gen_vec[i]
                    if (diff == 0): continue
                    if (diff == 0): 
                        out_str = "pixel not changed " + mdata.get_feature_names(i) + " " 
                    else:
                        out_str = "changed pixel " + mdata.get_feature_names(i) + " " 
                    out_str += "(idx: " + str(i) + "), "
                    out_str += "from value: " + str(src_vec[i]) + "; " 
                    out_str += "to value: " + str(gen_vec[i])
                    print(out_str)

    print("#7: Done!!")

        
if (__name__ == '__main__'):
    main()
