"""
UTILS FUNCTIONS TO HELP WITH VIS
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
import json

"""
modified from:
http://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html

turns a tree into a dict structure for JSON dump

tree_: scikit decision tree, can be extracted from random forest or decision tree models
feature_names: list of string names of features, in order of display in the df

out: dict as defined below
"""
def tree_to_dict(model, feature_names):
    tree_ = model.tree_
    n_nodes = tree_.node_count
    children_left = tree_.children_left
    children_right = tree_.children_right
    feature = tree_.feature
    threshold = tree_.threshold

    # traverse the tree
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    #aggregate results
    nodes = {}
    for i in range(n_nodes):
        nodes[i] = {
            "is_leaf": 1 if is_leaves[i] else 0,
            "depth": int(node_depth[i]),
            "id": int(i),
            "children_l": int(children_left[i]),
            "children_r": int(children_right[i]),
            "feature": feature_names[feature[i]],
            "threshold": threshold[i]
        }
    return {
        'n_nodes':int(n_nodes), 
        "nodes": nodes, 
        "feature_importances": list(model.feature_importances_)}

"""
modified from:
https://planspace.org/20151129-see_sklearn_trees_with_d3/

return decision tree rules in following format:
{name: "container thing",
 children: [{name: "leaf thing one"},
            {name: "leaf thing two"}]}

Structure of rules in a fit decision tree classifier

clf : DecisionTreeClassifier
    A tree that has already been fit.
features: list of str


"""
def rules(clf, features, node_index=0):
    node = {}
    if clf.tree_.children_left[node_index] == -1:  # indicates leaf
        num_labels = len(clf.tree_.value[node_index, 0])
        labels = list(range(num_labels))
        count_labels = zip(clf.tree_.value[node_index, 0], labels)
        node['name'] = ', '.join(('{} of {}'.format(int(count), label)
                                  for count, label in count_labels))
    else:
        feature = features[clf.tree_.feature[node_index]]
        threshold = clf.tree_.threshold[node_index]
        node['name'] = '{} > {}'.format(feature, threshold)
        left_index = clf.tree_.children_left[node_index]
        right_index = clf.tree_.children_right[node_index]
        node['children'] = [rules(clf, features, right_index),
                            rules(clf, features, left_index)]
    return node

"""
in: trained scikit random forest model, string list of feature names
out: list of dicts representation of trees for JSON dump
"""
def rf_to_dict(model, feature_names):
    trees = [clf for clf in model.estimators_]
    return {
        "feature_names":feature_names, 
        "trees":[tree_to_dict(tree,feature_names) for tree in trees], 
        "rules":[rules(tree, feature_names) for tree in trees]
    }

"""
creates json dump of RF data
in: trained scikit random forest model, string list of feature names, string of output file name
out: none

usage example:
utils.rf_json_dump(model, list(train_df), "test_rf_dump.json")
"""
def rf_json_dump(model, feature_names, out_file):
    data = rf_to_dict(model, feature_names)
    with open(out_file, 'w') as outfile:  
        json.dump(data, outfile)