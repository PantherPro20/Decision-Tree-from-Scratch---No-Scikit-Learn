import pandas as pd
import numpy as np


class Node:
    def __init__(self, features=None, threshold=None, left=None, right=None, value=None):
        self.features = features
        self.threshold= threshold
        self.left=left
        self.right=right

        #for leaf
        self.value=value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, max_depth=5, min_samples=10):
        self.max_depth=max_depth
        self.min_samples=min_samples

    def _gini_impurity(self, y):
        n_samples = len(y)
        _, counts = np.unique(y, return_counts=True)
        probality = counts / n_samples

        return 1 - np.sum(probality**2)

    def _split(self, X_column, split_threshold):

        left_ifxs = np.where(X_column <= split_threshold)[0]

        right_ifxs = np.where(X_column > split_threshold)[0]

        return left_ifxs, right_ifxs

    def _best_split(self, X, y):
        best_gini = float('inf')
        split_idx, split_thresh = None, None


        n_samples, n_features = X.shape

        for feat_idx in range(n_features):
            threshholds = np.unique(X[:, feat_idx])
            for thresh in threshholds:
                left_idxs, right_idxs = self._split(X[:, feat_idx], thresh)

                if len(left_idxs) == 0 or len(right_idxs) == 0:
                    continue


                #Calculate Gini Impurity for left and right for each iteration
                self._gini_impurity(y[left_idxs])
                self._gini_impurity(y[right_idxs])

                gini_left_right = (len(left_idxs) / n_samples) * (self._gini_impurity(y[left_idxs])) + (len(right_idxs) / n_samples) * (self._gini_impurity(y[right_idxs]))

                if gini_left_right < best_gini:
                    best_gini = gini_left_right
                    split_idx = feat_idx
                    split_thresh = thresh
                else:
                    continue
        return split_idx, split_thresh

    def fit(self, X,y ):
        n_samples, n_features = X.shape
        depth = 0
        self.root = self._build_tree(X,y, depth = 0)


    def _build_tree(self, X, y, depth = 0):
        n_samples, n_features = X.shape

        if depth >= self.max_depth or self._gini_impurity(y) == 0 or n_samples < self.min_samples_split or len(np.unique(y)) == 0:
            leaf_value = np.bincount(y).argmax()
            return Node(value=leaf_value)
        
        split_idx, split_thresh = self._best_split(X, y)
        left_idxs, right_idxs = self._split(X[:, split_idx], split_thresh)
        left_child = self._build_tree(X[left_idxs], y[left_idxs], depth + 1)
        right_child = self._build_tree(X[right_idxs], y[right_idxs], depth + 1)
        return Node(threshold=split_thresh,left=left_child,right=right_child, split_idx=split_idx)


    def _predict_row(self, node, x):
        if node.value is not None:
            return node.value
        if x[node.split_idx] <= node.threshold:
            return self._predict_row(node.left, x)
        else:
            return self._predict_row(node.right,x)

    def predict(self,X):
        predictions = []
        for row in X:
            pred = self._predict_row(self.root, row)
            predictions.append(pred)
        return np.array(predictions)

        





            
        

        
            
