# -*- coding: utf-8 -*-
# © 2018-2019 Nokia
#
#Licensed under the BSD 3 Clause license
#SPDX-License-Identifier: BSD-3-Clause



import os
import numpy as np
import time
import cv2
import csv
import pydotplus
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets.openml import fetch_openml
matplotlib.use('Agg')

from sklearn import tree, metrics
from sklearn.datasets import fetch_mldata
from sklearn.utils import Bunch
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib

from xgboost.sklearn import XGBClassifier, DMatrix
from xgboost import plot_tree
import scipy

from qtas import VeriGbError

# 
# def train_cnn_classifier_model(metadata, model_file, train_model=True, seed=None, batch_size=128, epochs=8, verb=2):
#     # needed in any case for the analysis
#     # np.random.seed(seed)
#     (x_train, y_train), (x_test, y_test) = (metadata.get_train_data(), metadata.get_train_label()), (metadata.get_test_data(), metadata.get_test_label()) 
#     
#     model = None
#     print("#" * 80)
#     if (not train_model):
#         print("Loading NN classifier from file.")
#         model = load_model(model_file)
#     else:
#         model = Sequential()
#         model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
#         model.add(Conv2D(64, (3, 3), activation='relu'))
#         model.add(MaxPooling2D(pool_size=(2, 2)))
#         model.add(Dropout(0.25))
#         model.add(Flatten())
#         model.add(Dense(128, activation='relu'))
#         model.add(Dropout(0.5))
#         model.add(Dense(metadata.get_num_labels(), activation='softmax'))
#         model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
#         
#         model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
#     
#         
#     print("- Analyzing model")
#     analyze_nn_model(model=model, obs_data=x_test, label_data=y_test)
#     
#     ## don't want redundant over writing (for the git...)
#     if (train_model): model.save(model_file)
#     return model


def train_classifier_model(metadata, model_file, model_type='skl', train_model=True, learn_r=0.1,  scale_pos_weight=1, esti_n=100, subs=1.0, maxd=3, seed=None, verb=2):
    model = None
    print("#" * 80)
    if (not train_model):
        print("Loading " + model_type.upper() + " gradient boosting classifier from file.")
        model = joblib.load(model_file)
    else:
        if (model_type.lower() == "xgb"):
            
            model = XGBClassifier(verbose=verb, seed=seed, subsample=subs, max_depth=maxd, n_estimators=esti_n, learning_rate=learn_r,
                                  scale_pos_weight=scale_pos_weight)
            print("Training XGB gradient boosting classifier.")
            model.fit(np.asarray(metadata.get_train_data()), metadata.get_train_label())
            #model.fit(np.asarray(metadata.get_train_data()), metadata.get_train_label(),
            #          eval_set=[(metadata.get_train_data(), metadata.get_train_label()),
            #                    (metadata.get_test_data(), metadata.get_test_label())])
        elif (model_type.lower() == "skl"):
            model = GradientBoostingClassifier(verbose=verb, random_state=seed, subsample=subs, max_depth=maxd,
                                                n_estimators=esti_n, learning_rate=learn_r)
            print("Training sklearn gradient boosting classifier.")
            model.fit(metadata.get_train_data(), metadata.get_train_label())
            # watchlist = [(mdata.get_train_data(), mdata.get_train_label()), (mdata.get_tset_data(), mdata.get_test_label())]
            # model.fit(mdata.get_train_data(), mdata.get_train_label(), verbose = verb, eval_set = watchlist)
        else:
            raise VeriGbError("unknown model type '" + model_type + "'")
            
    #
    print("- Analyzing model")
    analyze_model(model=model, metadata=metadata)
    
    ## don't want redundant over writing (for the git...)
    if (train_model): joblib.dump(model, model_file)
    return model


def analyze_model(model, metadata):
    t0 = time.time()
    results = {}
    predicted = np.array([])
    for i in range(0, len(metadata.get_test_data()), 128):  # go in chunks of size 128
        predicted_single = model.predict(metadata.get_test_data()[i:(i + 128)])
        predicted = np.append(predicted, predicted_single)
    results['conf_matrix'] = metrics.confusion_matrix(metadata.get_test_label(), predicted)
    results['accuracy'] = metrics.accuracy_score(metadata.get_test_label(), predicted)
    #results['score'] = model.score(metadata.get_test_data(), metadata.get_test_label())
    results['testing_time'] = time.time() - t0
    print("-- Classifier: Gradient Boosting")
    print("-- Testing time: %0.4fs" % results['testing_time'])
    print("-- Confusion matrix:\n%s" % results['conf_matrix'])
    print("-- Accuracy: %0.4f" % results['accuracy'])


# def analyze_nn_model(model, obs_data, label_data):
#     t0 = time.time()
#     results = {}
#     predicted = np.array([])
#     for i in range(0, len(obs_data), 128):  # go in chunks of size 128
#         predicted_single = model.predict(obs_data[i:(i + 128)])
#         if (len(predicted) == 0):
#             predicted = predicted_single 
#         else: 
#             predicted = np.vstack((predicted, predicted_single))
#     results['conf_matrix'] = metrics.confusion_matrix(label_data.argmax(axis=1), predicted.argmax(axis=1))
#     score = model.evaluate(obs_data, label_data, verbose=0)
#     results['loss'] = score[0]
#     results['accuracy'] = score[1]
#     results['testing_time'] = time.time() - t0
#     print("-- Classifier: CNN")
#     print("-- Testing time: %0.4fs" % results['testing_time'])
#     print("-- Confusion matrix:\n%s" % results['conf_matrix'])
#     print("-- Accuracy: %0.4f" % results['accuracy'])


# names: 'MNIST original', 'iris', 'leukemia', 'datasets-UCI iris', 'Whistler Daily Snowfall'
def my_fetch_mldata(dataname, data_home=None, load_ratio=1):
    return fetch_openml(dataname, data_home=data_home)


def export_image_example_to_pdf(file_name, src_image_vec, gen_image_vec, image_height, image_width, min_val, max_val,
                                image_scheme=1, image_flip=False, image_rot=False, cmap=None):
    fig = plt.figure(figsize=(8, 8))
    
    # changing the problem to float [0.0, 1.0]...
    if (min_val < 0):  # adding to all values " - min_val"
        src_image_vec = src_image_vec - min_val
        gen_image_vec = gen_image_vec - min_val
        max_val = max_val - min_val
        min_val = 0.0
    src_image_vec = src_image_vec * 1.0
    gen_image_vec = gen_image_vec * 1.0
    src_image_vec = src_image_vec / (max_val - min_val)
    gen_image_vec = gen_image_vec / (max_val - min_val)
    
    # generated
    fig.add_subplot(1, 4, 1)
    if (image_scheme == 1): gen_sol = gen_image_vec.reshape((image_height, image_width))
    else: gen_sol = gen_image_vec.reshape((image_height, image_width, image_scheme)) 
    if (image_flip): gen_sol = 1 - gen_sol
    if (image_rot): gen_sol = gen_sol.transpose()
    plt.imshow(gen_sol, vmin=0.0, vmax=1.0, cmap=cmap)

    # original
    fig.add_subplot(1, 4, 2)
    if (image_scheme == 1): src_sol = src_image_vec.reshape((image_height, image_width))
    else: src_sol = src_image_vec.reshape((image_height, image_width, image_scheme)) 
    if (image_flip): src_sol = 1 - src_sol
    if (image_rot): src_sol = src_sol.transpose()
    plt.imshow(src_sol, vmin=0.0, vmax=1.0, cmap=cmap)

    # delta
    fig.add_subplot(1, 4, 3)
    diff_sol = np.absolute(gen_sol - src_sol)
    if (image_flip): diff_sol = 1 - diff_sol
    if (image_rot): diff_sol = diff_sol.transpose()
    plt.imshow(diff_sol, vmin=0.0, vmax=1.0, cmap=cmap)
    
    # scaling manually (flapping back and forth)
    if (image_flip): diff_sol = 1 - diff_sol
    diff_sol = diff_sol * (1 / np.max(diff_sol))
    if (image_flip): diff_sol = 1 - diff_sol
        
    fig.add_subplot(1, 4, 4)
    plt.imshow(diff_sol, vmin=0.0, vmax=1.0, cmap=cmap)
    
    # writing to file
    plt.savefig(file_name)
    plt.close(fig)


def export_trees_to_pdf(folder_name, model, classes_to_print, num_of_classes):
    if (isinstance(model, GradientBoostingClassifier)):
        for class_idx in classes_to_print:
            for tree_idx in range(len(model.estimators_)): 
                sub_tree = model.estimators_[tree_idx, class_idx]
                dot_data = tree.export_graphviz(sub_tree,
                    out_file=None, filled=True, rounded=True,
                    special_characters=True, proportion=True
                )
                graph = pydotplus.graph_from_dot_data(dot_data)  
                graph.write_pdf(folder_name + "/L" + str(class_idx) + "tree" + str(tree_idx) + ".pdf") 
    elif(isinstance(model, XGBClassifier)):
        for class_idx in classes_to_print:
            for tree_idx in range(model.n_estimators):
                plot_tree(model, num_trees=(tree_idx * num_of_classes + class_idx), rankdir='LR')
                fig = matplotlib.pyplot.gcf()
                fig.set_size_inches(15, 8)
                plt.savefig(folder_name + "/L" + str(class_idx) + "tree" + str(tree_idx) + ".png")
            plt.close()
    else:
        raise VeriGbError("unknown model type '" + str(type(model)) + "'")



