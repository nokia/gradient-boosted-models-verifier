# -*- coding: utf-8 -*-
# © 2018-2019 Nokia
#
#Licensed under the BSD 3 Clause license
#SPDX-License-Identifier: BSD-3-Clause



import re

#==================================
# parsing xgboost textual output


class XGBTree:

    class Node:

        def __init__(self, node_id, node_cond=None, is_leaf=False) :
            self.is_leaf = is_leaf
            self.node_id = node_id  # we need it as an int
            self.node_cond = node_cond
            self.true_child = None
            self.false_child = None
            
        def __str__(self):
            if (self.is_leaf):
                return str(self.node_id) + " " + self.node_cond + "\n"
            # else: internal
            ret = str(self.node_id) + ": " + "[" + self.node_cond + "]" 
            ret += " yes=" + str(self.true_child.node_id) 
            ret += ", no=" + str(self.false_child.node_id) + "\n"
            ret += str(self.true_child) + str(self.false_child)
            return ret

    def __init__(self) :
        self.nodes = dict()
        self.edges = dict()
        self.feature = dict()
        self.threshold = dict()
        self.false_children = dict()
        self.true_children = dict()
        
    def __str__(self):
        return str(self.nodes.get(0))
    
    def get_node(self, node_id):
        node = self.nodes.get(node_id)
        if node is None:
            node = XGBTree.Node(node_id)
            self.nodes[node_id] = node
        return node

    def add_node(self, node_id, node_cond, is_leaf):
        node = self.get_node(node_id)
        if (node_cond != None):  # just in case...
            # TODO: check that the second call doesn't overrides these values...
            node.node_cond = node_cond
            node.is_leaf = is_leaf
        
        if (is_leaf == False) :
            match = _FEATPAT.match(node_cond)
            self.feature[node_id] = int(match.group(1))
            self.threshold[node_id] = float(match.group(2))
        else:
            match = _LEAFEVALPAT.match(node_cond)
            self.feature[node_id] = None
            self.threshold[node_id] = float(match.group(1))
           
        return node
    
    def add_true_edge(self, from_node, to_node_id):
        to_node = self.get_node(to_node_id)
        from_node.true_child = to_node
        self.true_children[from_node.node_id] = to_node.node_id
        
    def add_false_edge(self, from_node, to_node_id):
        to_node = self.get_node(to_node_id)
        from_node.false_child = to_node
        self.false_children[from_node.node_id] = to_node.node_id
        
    #===========================================================================
    # def eval(self, input_str):
    #     node = self.nodes.get(0)
    #     return self.eval_rec(node, input_str)
    # 
    # def eval_rec(self, node, inp):
    #     if (node.is_leaf):
    #         return self.threshold[node.node_id]
    #     else:
    #         feat_val_in_input = inp[self.feature[node.node_id]]
    #         feat_val_in_cond = self.threshold[node.node_id]
    #         if (feat_val_in_input < feat_val_in_cond):
    #             return self.eval_rec(node.true_child, inp)
    #         else:
    #             return self.eval_rec(node.false_child, inp)
    #===========================================================================
    

def parse_trees(model): 
    # trees = model.get_dump(fmap='')
    # trees = model.get_booster().get_dump(fmap='', dump_format='json')
    trees = model.get_booster().get_dump(fmap='')
    parsed_trees = []
    for t in trees:
        parsed_trees.append(__parse_single_tree(t))
    return parsed_trees

    
def __parse_single_tree(treeStr):
    tree = XGBTree()
    line = treeStr.split()

    for i, text in enumerate(line):
        if (text[0].isdigit()):
            node = __parse_node(tree, text)
        elif (i == 0):  # 1st string must be node
            raise ValueError('Unable to parse given string as tree')
        else:
            __parse_edge(tree, node, text)
    
    return tree


_NODEPAT = re.compile(r'(\d+):\[(.+)\]')
_LEAFPAT = re.compile(r'(\d+):(leaf=.+)')
_EDGEPAT = re.compile(r'yes=(\d+),no=(\d+),missing=(\d+)')
_EDGEPAT2 = re.compile(r'yes=(\d+),no=(\d+)')
_FEATPAT = re.compile(r'f(\d+)<(.+)')
_LEAFEVALPAT = re.compile(r'leaf=(.+)')


def __parse_node(tree, text):
    """parse dumped node"""
    match = _NODEPAT.match(text)
    if (match is not None):  # internal node
        node_id = int(match.group(1))
        node_cond = match.group(2)
        node = tree.add_node(node_id, node_cond, False)
        return node
    match = _LEAFPAT.match(text)
    if (match is not None):  # leaf node
        node_id = int(match.group(1))
        node_cond = match.group(2)
        node = tree.add_node(node_id, node_cond, True)
        return node
    raise ValueError('Unable to parse node: {0}'.format(text))


def __parse_edge(tree, node, text):
    """parse dumped edge"""
    try:
        match = _EDGEPAT.match(text)
        if (match is not None):
            yes_id_str, no_id_str, missing = match.groups()  # @UnusedVariable
            tree.add_true_edge(node, int(yes_id_str))
            tree.add_false_edge(node, int(no_id_str))
            return
    except ValueError:
        # trying a different pattern
        match = _EDGEPAT2.match(text)
        if match is not None:
            yes_id_str, no_id_str = match.groups()
            tree.add_true_edge(node, int(yes_id_str))
            tree.add_false_edge(node, int(no_id_str))
            return
    
    raise ValueError('Unable to parse edge: {0}'.format(text))
   
