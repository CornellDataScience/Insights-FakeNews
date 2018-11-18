from c45 import *

class C45Tree():
	"""
	C4.5 Decision Tree
	"""
    def __init__(self, criterion = entropy, max_depth = None):
        self.classifier = None
        self.criterion = criterion
        #self.fitted = False

    def fit(self, X):
        """
        X is the set of data where the last column of data is labels.
        """
        self.classifier = train(X, self.criterion)
        #self.fitted = True

        def prune(self):
            prune_tree(self.classifier, 0.5, self.criterion)

        prune(self)

    def classify(self, x):
        """
        x is the set of values to be classified.
        """
        return predict(x, self.classifier)
