"""
C4.5 Binary Decision Tree Implementation

Usage: 
Read csv file in; will be stored as a 2 Dimensional list. (See fread())
Train a classifier (i.e. train(list))
Prune the decision tree (i.e. prune_tree(tree, 0.5))
Predict the result (i.e. predict([.....], classifier))

The function assumes that the last column of your data is populated by labels.

Example of usage:

data = fread("./test_val_dump.csv", True)
drop_first_col = []
for x in data:
    drop_first_col.append(x[1:])
tree = train(drop_first_col)
prune_tree(tree, 0.5)
print(predict([2,0,2,6,1,0,0,3,1,0,0,2,0,0,0,0,0,1,.223606798,0,.285714,.141421,0,.253546],tree))

"""

from collections import OrderedDict
from math import log
import csv

def n_distinct_dict(rows):
    counts = dict()
    for x in rows:
        xs = x[-1]
        if xs not in counts: 
            #Add it to the counts dictionary
            counts[xs] = 0
        counts[xs] += 1
    return counts

def entropy(X):
    """
    Calculate Entropy (as per Octavian)
    """
    counts = n_distinct_dict(X)
    log_2 = lambda x: log(x)/log(2)
    #Declare entropy value
    entropy = 0.0
    
    for c in counts:
        #Calculate P(C_i)
        p = float(counts[c])/len(X)
        entropy = entropy -  p*log_2(p)
    return entropy

def gini(X):
    """
    Calculate Gini Index
    """
    total = len(X)
    counts = n_distinct_dict(X)
    imp = 0.0
    
    for k1 in counts:
        p1 = float(counts[k1])/total  
        for k2 in counts:
            if k1 == k2: continue
            p2 = float(counts[k2])/total
            imp += p1*p2
    return imp

class Tree:
    """
    Decision Tree class
    """
    def __init__(self, col=-1, value=None, right_branch=None, left_branch=None, results=None):
        self.col = col
        self.value = value
        self.right_branch = right_branch
        self.left_branch = left_branch
        self.results = results


def prune_tree(tree, least_gain, eval_fun = entropy):
    """
    tree : type Tree
    eval_fun : entropy(X) or gini(X)
    least_gain : float
    """
    
    if tree.right_branch.results == None: #if the right branch is a node
        prune_tree(tree.right_branch, least_gain, eval_fun)
    if tree.left_branch.results == None: #if the left branch is a node
        prune_tree(tree.left_branch, least_gain, eval_fun)
    if (tree.right_branch.results != None) and (tree.left_branch.results != None):
        right, left = [], []
        for v, c in tree.right_branch.results.items(): 
            right += [[v]] * c
        for v, c in tree.left_branch.results.items(): 
            left += [[v]] * c
        p = float(len(right)) / len(left + right)
        diff_entropy = eval_fun(right+left) - p*eval_fun(right) - (1-p)*eval_fun(left)
        if diff_entropy < least_gain:
            tree.right_branch, tree.left_branch = None, None
            tree.results = n_distinct_dict(left + right)

"""
Helper functions: type_conversion, fread
"""
def type_conversion(val):
        val = val.strip()
        
        try:
            if '.' in val:
                return float(val)
            else:
                return int(val)
            
        except ValueError:
            #For other types, return
            return val
        
def fread(f, col_labels = False):
    """
    takes a filepath, f, and a boolean argument, col_labels.
    By default, col_labels is False, implying that the columns do not have labels. If set to true,
    fread will remove the row containing column labels at the end.
    """
    data = csv.reader(open(f, 'rt'))
    lst = [[type_conversion(i) for i in r] for r in data]
    if col_labels:
        lst.pop(0)
    return lst


def partition(r, c, val):
    """
    Function to partition the data based on value
    """
    #Declare anonymous function
    split_fun = None
    if isinstance(val, float) or isinstance(val, int): 
        #Anonymous function for numeric values
        split_fun = lambda row : row[c] >= val
    else: 
        #For string values
        split_fun = lambda row : row[c] == val
    list1 = [row for row in r if split_fun(row)]
    list2 = [row for row in r if not split_fun(row)]
    return (list1, list2)



def train(lst, criteria = entropy):
    """
    Decision tree construction - by default, the entropy function is used to calculate the criteria for splitting. 
    lst : dataframe with the last column reserved for labels
    criteria : entropy or gini calculation function
    """
    #Base Case: Empty Set
    if len(lst) == 0: 
        return Tree()
    
    #Calculate Entropy/Gini of current X, declare A_best, create sets/gain accordingly
    score = criteria(lst)
    Attribute_best = None
    Set_best = None
    Gain_best = 0.0
    

    num_col = len(lst[0]) - 1  # last column of lst is labels
    for c in range(num_col):
        col_val = [row[c] for row in lst]
        for value in col_val:
            #Split dataset
            (set1, set2) = partition(lst, c, value)
            # Calculate Gain
            p = float(len(set1))/len(lst)
            gain = score - p*criteria(set1) - (1-p)*criteria(set2)
            if gain>Gain_best and len(set1)>0 and len(set2)>0:
                Gain_best = gain
                Attribute_best = (c, value)
                Set_best = (set1, set2)

    if Gain_best > 0:
        #Recursive Call on partitioned Sets
        right_branch = train(Set_best[0])
        left_branch = train(Set_best[1])
        return Tree(col=Attribute_best[0], value=Attribute_best[1], right_branch=right_branch, left_branch=left_branch)
    
    else:
        return Tree(results=n_distinct_dict(lst))


def tree_classify(X, tree):
    """
    Classification function using a list read from fread, X, and grown Decision Tree, tree
    """
    if tree.results != None:
        return (tree.results)
    else:
        b = None
        val = X[tree.col] #Retrieve label from dataframe X
        if isinstance(val, float) or isinstance(val,int):
            #Traversing decision tree for numerics
            if val >= tree.value:
                branch = tree.right_branch
            else:
                branch = tree.left_branch
            
        else:
            #Traversing decision tree for non-numeric types
            if val == tree.value:
                branch = tree.right_branch
            else:
                branch = tree.left_branch
    return tree_classify(X, branch)

def predict(x, classifier):
    return list(tree_classify(x, classifier))[0]