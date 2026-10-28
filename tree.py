import numpy as np

class Node:
    def __init__(self, split_idx=None, threshold=None, left=None, right=None, value=None):
        self.split_idx = split_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # For leaf nodes

    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def _gini_impurity(self, y):
        n_samples = len(y)
        if n_samples == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        prob = counts / n_samples
        return 1 - np.sum(prob**2)

    def _split(self, X_column, split_threshold):
        left_idxs = np.where(X_column <= split_threshold)[0]
        right_idxs = np.where(X_column > split_threshold)[0]
        return left_idxs, right_idxs

    def _best_split(self, X, y):
        best_gini = float('inf')
        split_idx, split_thresh = None, None

        n_samples, n_features = X.shape

        for feat_idx in range(n_features):
            thresholds = np.unique(X[:, feat_idx])
            for thresh in thresholds:
                left_idxs, right_idxs = self._split(X[:, feat_idx], thresh)

                if len(left_idxs) == 0 or len(right_idxs) == 0:
                    continue

                gini_left_right = (len(left_idxs) / n_samples) * self._gini_impurity(y[left_idxs]) + \
                                  (len(right_idxs) / n_samples) * self._gini_impurity(y[right_idxs])

                if gini_left_right < best_gini:
                    best_gini = gini_left_right
                    split_idx = feat_idx
                    split_thresh = thresh

        return split_idx, split_thresh

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape

        # Stopping criteria
        if (depth >= self.max_depth or 
            self._gini_impurity(y) == 0 or 
            n_samples < self.min_samples_split):
            leaf_value = np.bincount(y).argmax()
            return Node(value=leaf_value)
        
        split_idx, split_thresh = self._best_split(X, y)
        
        # If no valid split was found, default to a leaf node
        if split_idx is None:
            leaf_value = np.bincount(y).argmax()
            return Node(value=leaf_value)

        left_idxs, right_idxs = self._split(X[:, split_idx], split_thresh)
        
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            leaf_value = np.bincount(y).argmax()
            return Node(value=leaf_value)

        left_child = self._build_tree(X[left_idxs], y[left_idxs], depth + 1)
        right_child = self._build_tree(X[right_idxs], y[right_idxs], depth + 1)
        
        return Node(threshold=split_thresh, left=left_child, right=right_child, split_idx=split_idx)

    def _predict_row(self, node, x):
        if node.is_leaf_node():
            return node.value
        if x[node.split_idx] <= node.threshold:
            return self._predict_row(node.left, x)
        else:
            return self._predict_row(node.right, x)

    def predict(self, X):
        predictions = []
        for row in X:
            pred = self._predict_row(self.root, row)
            predictions.append(pred)
        return np.array(predictions)