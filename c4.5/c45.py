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

from collections import OrderedDict, Counter
from math import log
import csv
from random import randrange


def entropy(X):
    """
    Calculate Entropy (as per Octavian)
    """
    counts = Counter([x[-1] for x in X])
    n_rows = len(X)
    # Declare entropy value
    entropy = 0.0

    for c in counts:
        # Calculate P(C_i)
        p = float(counts[c])/n_rows
        entropy -= p*log(p)/log(2)
    return entropy


def gini(X):
    """
    Calculate Gini Index
    """
    n_rows = len(X)
    counts = Counter([x[-1] for x in X])
    imp = 0.0

    for k1 in counts:
        p1 = float(counts[k1])/n_rows
        for k2 in counts:
            if k1 == k2: continue
            p2 = float(counts[k2])/n_rows
            imp += p1*p2
    return imp


class Tree:
    """
    Decision Tree class
    """

    def __init__(self, feature=-1, value=None, right_branch=None, left_branch=None, results=None, gain = 0.0):
        self.feature = feature
        self.value = value
        self.right_branch = right_branch
        self.left_branch = left_branch
        self.results = results
        self.gain = gain


def prune_tree(tree, least_gain, eval_fun=entropy):
    """
    tree : type Tree
    eval_fun : entropy(X) or gini(X)
    least_gain : float
    """

    if tree.right_branch.results == None:  # if the right branch is a node
        prune_tree(tree.right_branch, least_gain, eval_fun)
    if tree.left_branch.results == None:  # if the left branch is a node
        prune_tree(tree.left_branch, least_gain, eval_fun)
    if (tree.right_branch.results != None) and (tree.left_branch.results != None):
        right, left = [], []
        for v, c in tree.right_branch.results.items():
            right += [[v]] * c
        for v, c in tree.left_branch.results.items():
            left += [[v]] * c
        p = float(len(right)) / len(left + right)
        diff_entropy = eval_fun(right+left) - p * \
                                eval_fun(right) - (1-p)*eval_fun(left)
        if diff_entropy < least_gain:
            tree.right_branch, tree.left_branch = None, None
            tree.results = Counter([x[-1] for x in (left+right)])


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
            # For other types, return
            return val


def fread(f, col_labels=False):
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


def train(lst, depth=0, max_depth=100, min_samples_leaf=1, min_samples_split=2, criteria=entropy):
    """
    Decision tree construction - by default, the entropy function is used to calculate the criteria for splitting.
    lst : dataframe with the last column reserved for labels
    criteria : entropy or gini calculation function
    """
    # Base Case: Empty Set
    if len(lst) == 0:
        return Tree()
    elif len(lst) > min_samples_split:
        # Calculate Entropy/Gini of current X, declare A_best, create sets/gain accordingly
        score = criteria(lst)
        Attribute_best = None
        Set_best = None
        Gain_best = 0.0

        num_col = len(lst[0]) - 1  # last column of lst is labels

        for c in range(num_col):
            col_val = list(sorted(set([row[c] for row in lst])))
            for value in col_val:
                # Partition Dataset
                if isinstance(value, float) or isinstance(value, int):  # numerics
                    set1 = [row for row in lst if row[c] >= value]
                    set2 = [row for row in lst if row[c] < value]
                else:  # strings
                    set1 = [row for row in lst if row[c] == value]
                    set2 = [row for row in lst if row[c] != value]
                if len(set1) > min_samples_leaf and len(set2) > min_samples_leaf: #check that leaves are large enough
                    # Calculate Gain
                    p = float(len(set1))/len(lst)
                    gain = score - p*criteria(set1) - (1-p)*criteria(set2)
                    if gain > Gain_best:
                        Gain_best = gain
                        Attribute_best = (c, value)
                        Set_best = (set1, set2)

        if Gain_best > 0 and depth < max_depth: #check max depth
            # Recursive Call on partitioned Sets
            r = train(Set_best[0], depth+1, max_depth)
            l = train(Set_best[1], depth+1, max_depth)
            return Tree(feature=Attribute_best[0], value=Attribute_best[1], right_branch=r, left_branch=l, gain = Gain_best)
        else:
            return Tree(results=Counter([x[-1] for x in lst]), gain = Gain_best)
    else: #partition is too small to split
        return Tree(results=Counter([x[-1] for x in lst]))


def tree_classify(X, tree):
    """
    Classification function using a list read from fread, X, and grown Decision Tree, tree
    """
    if tree.results != None:
        return (tree.results)
    else:
        val = X[tree.feature]  # Retrieve label from dataframe X
        if isinstance(val, float) or isinstance(val, int):
            # Traversing decision tree for numerics
            if val >= tree.value:
                branch = tree.right_branch
            else:
                branch = tree.left_branch

        else:
            # Traversing decision tree for non-numeric types
            if val == tree.value:
                branch = tree.right_branch
            else:
                branch = tree.left_branch
    return tree_classify(X, branch)


def predict(x, classifier):
    return tree_classify(x, classifier).most_common(1)[0][0]



"""
usage:

tree = DecisionTree()
tree.train(data)
tree.classify(input)

params:
criterion
max_depth - default 100
min_samples_leaf - default 1
min_samples_split - default 2
"""

class DecisionTree():
    """
    Decision Tree Class
    """
    def __init__(self, **kwargs):
        self.classifier = None
        self.criterion = kwargs.get('criterion', entropy)
        self.max_depth = kwargs.get('max_depth', 100)
        self.min_samples_leaf = kwargs.get('min_samples_leaf', 1)
        self.min_samples_split = kwargs.get('min_samples_split', 2)

    def fit(self, X):
        """
        X is the set of data where the last column of data is labels.
        """
        self.classifier = train(X, 0, self.max_depth, self.min_samples_leaf, self.min_samples_split, self.criterion)
        prune_tree(self.classifier, 0.5, self.criterion)

    def classify(self, x):
        """
        x is the set of values to be classified.
        """
        return predict(x, self.classifier)

#Bagging tree prediction
def bag(treelst, row):
    predictions = [predict(row, tree.classifier) for tree in treelst]
    return max(set(predictions), key=predictions.count)

def subsample(X, size):
    sample = []
    n = round(len(X) * size)
    while len(sample) < n:
        index = randrange(len(X))
        sample.append(X[index])
    return sample

def create_forest(X, max_depth, subsample_ratio, min_samples_leaf = 1, min_samples_split = 2, n_trees = 250):
    """
    Create a list containing decision trees fitted to a subsample of the data.
    max_depth : max tree depth
    subsample_ratio : subsample ratio
    min_samples_leaf : minimum samples per tree leaf
    min_samples_split : minimum samples per split
    n_trees : number of decision trees to create
    """
    forest = []
    for i in range(n_trees):
        sample = subsample(X, subsample_ratio)
        dt = DecisionTree(max_depth = max_depth, min_samples_leaf = min_samples_leaf, min_samples_split = min_samples_split)
        dt.fit(sample)
        forest.append(dt)
    return forest

def forest_predictions(X, forest):
    """
    List of predictions from a random forest.
    """
    pred = [bag(forest, r) for r in X]
    return(pred)

def predict_probabilities(forest, inputs):
    """
    forest is a list of DecisionTree objects.
    """
    if(forest == None):
        raise Exception("Random Forest has not been fitted. Please call fit() first.")
    prob0 = 0
    prob1 = 0
    for i in range(len(forest)):
        temp0 = prob0 * (i)
        temp1 = prob1 * (i)
        classifications = tree_classify(inputs, forest[i].classifier)
        prob0 = (temp0 + (classifications[0] / (classifications[0] + classifications[1]))) / (i + 1)
        prob1 = (temp1 + (classifications[1] / (classifications[0] + classifications[1]))) / (i + 1)
    if prob0>=prob1:
        #If 0 is at least as likely as 1, return 0 for deterministic behavior.
        return (0, prob0)
    else:
        return (1, prob1)
    #else:
        #return (0, prob0)

"""
The following two functions are used to evaluate a RandomForest. They have been modified from Jason Brownlee's implementation, along with the score function.
"""
# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

class RandomForest():
    """
    Random Forest Classifier
    """
    def __init__(self, **kwargs):
        self.forest = None
        self.subsample_ratio = kwargs.get('subsample_ratio')
        self.criterion = kwargs.get('criterion', entropy)
        self.n_trees = kwargs.get('n_trees', 25)
        self.max_depth = kwargs.get('max_depth', 100)
        self.min_samples_leaf = kwargs.get('min_samples_leaf', 1)
        self.min_samples_split = kwargs.get('min_samples_split', 2)


    def fit(self, X):
        self.forest = create_forest(X, self.max_depth, self.subsample_ratio, self.min_samples_leaf, self.min_samples_split, self.n_trees)

    def predictions(self, X):
        """
        predictions is the list containing the prediction of each DecisionTree in the forest.
        """
        if(self.forest == None):
            raise Exception("Random Forest has not been fitted. Please call fit() first.")
        else:
            return forest_predictions(X, self.forest)

    def predict_probs(self, X):
        """
        predict_probs is the (prediction, probability) tuple representing the result of the RandomForest object.
        """
        return predict_probabilities(self.forest, X)

    def predict(self, X):
        """
        predict is the int (0,1) representing the label classified by the RandomForest object.
        """
        return predict_probabilities(self.forest, X)[0]
    """
    def score(self, data, n_folds = 5):
        folds = cross_validation_split(data, n_folds)
        scores = list()
        for fold in folds:
            train_set = list(folds)
            train_set.remove(fold)
            train_set = sum(train_set, [])
            test_set = list()
            for row in fold:
                row_copy = list(row)
                test_set.append(row_copy)
                row_copy[-1] = None
                rf = RandomForest(subsample_ratio = self.subsample_ratio, criterion = self.criterion, n_trees = self.n_trees, max_depth = self.max_depth, min_samples_leaf = self.min_samples_leaf, min_samples_split = self.min_samples_split)
                rf.fit(train_set)
                predicted = rf.predictions(test_set)
                print(rf.predictions(test_set))
                actual = [row[-1] for row in fold]
                accuracy = accuracy_metric(actual, predicted)
                scores.append(accuracy)
            return (sum(scores)/float(len(scores)))
    """
# Evaluate an algorithm using a cross validation split
def evaluate_forest(dataset, n_folds, subsample_ratio, ntrees = 25, maxdepth = 100, minsamplesleaf = 1, minsamplessplit = 2):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        rf = RandomForest(subsample_ratio = subsample_ratio, criterion = entropy, n_trees = ntrees, max_depth = maxdepth, min_samples_leaf = minsamplesleaf, min_samples_split = minsamplessplit)
        rf.fit(train_set)
        predicted = rf.predictions(test_set)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores

def mean_accuracy(dataset, n_folds, subsample_ratio, ntrees = 25, maxdepth = 100, minsamplesleaf = 1, minsamplessplit = 2):
    scores = evaluate_forest(dataset, n_folds, subsample_ratio, ntrees = 25, maxdepth = 100, minsamplesleaf = 1, minsamplessplit = 2)
    return (sum(scores)/float(len(scores)))

def evaluate_tree(dataset, n_folds):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        dt = DecisionTree()
        dt.fit(train_set)
        predicted = [dt.classify(x) for x in test_set]
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores

def mean_tree_accuracy(dataset, n_folds):
    scores = evaluate_tree(dataset, n_folds)
    return (sum(scores)/float(len(scores)))
