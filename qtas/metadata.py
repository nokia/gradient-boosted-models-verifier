# -*- coding: utf-8 -*-
# © 2018-2019 Nokia
#
#Licensed under the BSD 3 Clause license
#SPDX-License-Identifier: BSD-3-Clause


from sklearn.model_selection import train_test_split
import numpy as np
# from z3 import *  # @UnusedWildImport

from . import VeriGbError


class Metadata():
    
    def __init__(self, train_obs, test_obs, train_label, test_label):
        self.train_obs = train_obs 
        self.test_obs = test_obs
        self.train_label = train_label
        self.test_label = test_label
        #self.labels_type = np.unique(np.unique(self.train_label) + np.unique(self.test_label))  
        self.labels_type = np.unique(self.train_label)  
         
        self.feature_names = None
        self.image_height = None
        self.image_width = None
    
    @classmethod
    def raw_data_constructor(cls, data, labels, shuffle=True, test_size=0.25, seed=0):
        train_obs, test_obs, train_label, test_label = train_test_split(
            data, labels, shuffle=shuffle, test_size=test_size, random_state=seed
        )
        return cls(train_obs, test_obs, train_label, test_label)
    
#     def make_reserved_metadata(self, reserved_tests):    
#         reserved_ratio = 1.0 - 1.0 * reserved_tests / len(self.get_test_label())
#         ret = Metadata.raw_data_constructor(
#             data=self.get_test_data(), labels=self.get_test_label(),
#             shuffle=self.shuffle, test_size=reserved_ratio, seed=self.seed
#         )
#         
#         # updating self to remove the reserved
#         self.data_dict['test']['X'] = ret.data_dict['test']['X']
#         self.data_dict['test']['y'] = ret.data_dict['test']['y']
#         return ret
#     
#     # not really needed, but I prefer the imports, dependencies and mess to be here...    
#     @staticmethod
#     def split_reserved(raw, reserved_tests, shuffle, seed):
#         reserved_ratio = 1.0 - 1.0 * reserved_tests / len(raw.labels)
#         reserved_obs, raw_obs, reserved_label, raw_label = train_test_split(
#             raw.data, raw.labels, shuffle=shuffle, test_size=reserved_ratio, random_state=seed
#         )
#         return reserved_obs, raw_obs, reserved_label, raw_label 

    def get_image_idx_per_label(self, num_of_samples_per_label=1, seed=None):
        per_label_map = {}
        # for idx in np.random.RandomState(seed=seed).permutation(range(len(self.test_obs))):
        for idx in np.random.RandomState(seed=seed).permutation(len(self.test_obs)):
            label = int(self.test_label[idx])
            # if the label list does not exits yet -- create it
            if (label not in per_label_map):
                per_label_map[label] = []
            # if collected enough -- continue
            if (len(per_label_map[label]) >= num_of_samples_per_label):
                continue
            per_label_map[label].append(idx)
        return per_label_map
    
    def get_train_data(self, train_idx=None):
        if (train_idx == None):
            return self.train_obs
        return self.train_obs[train_idx]

    def get_train_label(self, train_idx=None):
        if (train_idx == None):
            return self.train_label
        return self.train_label[train_idx]
    
    def get_train_len(self):
        return len(self.get_train_label())
        
    def get_test_data(self, test_idx=None):
        if (test_idx == None):
            return self.test_obs
        return self.test_obs[test_idx]

    def get_test_label(self, test_idx=None):
        if (test_idx == None):
            return self.test_label
        return self.test_label[test_idx]
    
    def get_test_len(self):
        return len(self.get_test_label())
        
    def get_num_features(self):
        return len(self.get_train_data(0))
    
    def get_num_labels(self):
        return len(self.labels_type)
    
    def get_labels(self):
        return self.labels_type

    def get_feature_names(self, idx=None):
        if (self.feature_names == None):
            self.set_naive_feature_names()
        return self.feature_names if (idx == None) else self.feature_names[idx] 
    
    def set_naive_feature_names(self):
        self.feature_names = []
        for i in range(self.get_num_features()):
            self.feature_names.append(str(i))
        #
        
    def set_image_feature_names(self, image_height=None, image_width=None, image_scheme=1):
        num_of_pixels = self.get_num_features() / image_scheme
        if ((image_height == None) and (image_width == None)):
            self.image_height = int(num_of_pixels ** (0.5))
            self.image_width = self.image_height
        elif (image_height == None):  # only height is None
            self.image_height = num_of_pixels / image_width
            self.image_width = image_width
        elif (image_width == None):  # only width is None
            self.image_width = num_of_pixels / image_height
            self.image_height = image_height
        if (self.image_height * self.image_width != num_of_pixels):
            raise VeriGbError("Problem naming features (wrong height-width)")
        #
        self.feature_names = []
        for i in range(self.image_height):
            for j in range(self.image_width):
                for c in range(image_scheme):
                    self.feature_names.append('pR' + str(i) + 'C' + str(j) + 's' + str(c))
        #
    
    def get_image_height(self):
        '''
        Can return None if set_image_feature_names was not called first
        '''
        return self.image_height
        
    def get_image_width(self):
        '''
        Can return None if set_image_feature_names was not called first
        '''
        return self.image_width
        

    def get_per_label_count(self, label=None):
        unique, counts = np.unique(self.train_label, return_counts=True)
        if (label == None):
            return unique, counts
        else:
            for idx in range(0, len(unique)):
                if (label ==  unique[idx]):
                    return counts[idx]
                
                
    def get_train_or_test_data_per_label(self, dataset, num_of_samples_per_label=1):
        per_label_map = {}
        if (dataset == 'train'):
            data = self.train_obs
            labels = self.train_label
        else:
            data = self.test_obs
            labels = self.test_label
        
        for idx in range(len(data)):
            val = data[idx]
            label = labels[idx]
            label = int(label)
            
            # if the label list does not exits yet -- create it
            if (not (label in per_label_map)): #python 3.6
            #if (not per_label_map.has_key(label)): #python 2.7
                per_label_map[label] = []
            # if collected enough -- continue
            if (len(per_label_map[label]) >= num_of_samples_per_label):
                continue
            
            per_label_map[label].append(val)
            
        for label in per_label_map.keys():
            per_label_map[label] = np.array(per_label_map[label])
            
        return per_label_map
